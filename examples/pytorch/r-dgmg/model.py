import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.distributions import Bernoulli, Categorical
import copy


class Metagraph():
    def __init__(self, canonical_etypes):
        self.canonical_etypes = canonical_etypes
        self.canonical_etype_to_index = {}
        self.index_to_canonical_etype = {}
        self.ntype_to_index = {}
        self.index_to_ntype = {}
        self.nodes = []
        relations = {}
        num_nodes = {}
        edge_index = 1
        node_index = 1
        for relation in self.canonical_etypes:
            relations[relation] = ([], [])
            num_nodes[relation[0]] = 0
            num_nodes[relation[2]] = 0
            self.canonical_etype_to_index[relation] = edge_index
            self.index_to_canonical_etype[edge_index] = relation
            edge_index += 1
            if relation[0] not in self.ntype_to_index:
                self.ntype_to_index[relation[0]] = node_index
                self.index_to_ntype[node_index] = relation[0]
                node_index += 1
            if relation[2] not in self.ntype_to_index:
                self.ntype_to_index[relation[2]] = node_index
                self.index_to_ntype[node_index] = relation[2]
                node_index += 1
            if relation[0] not in self.nodes: self.nodes.append(relation[0])
            if relation[2] not in self.nodes: self.nodes.append(relation[2])
        self.empty_graph = dgl.heterograph(relations, num_nodes_dict=num_nodes)


# Multi-relational graph readout function
class GraphEmbed(nn.Module):
    def __init__(self, node_hidden_size, metagraph):
        super(GraphEmbed, self).__init__()

        # Setting from the paper
        self.graph_hidden_size = 2 * node_hidden_size

        # Embed graphs
        self.metagraph = metagraph
        self.node_gatings = nn.ModuleDict()
        self.node_to_graphs = nn.ModuleDict()

        # We need to loop through the node types; it's sequential
        # I don't know whether it's possible to parallelize it
        for metagraph_node in self.metagraph.nodes:
            self.node_gatings[metagraph_node] = nn.Sequential(
                # node_hidden_size is the same now for all nodes types
                nn.Linear(node_hidden_size, 1),
                nn.Sigmoid()
            )
            self.node_to_graphs[metagraph_node] = nn.Linear(node_hidden_size,
                                                            self.graph_hidden_size)

    def forward(self, g):
        if g.number_of_nodes() == 0:
            return torch.zeros(1, self.graph_hidden_size)
        else:
            # Node features are stored as hv in ndata.
            embeddings = []
            for metagraph_node in self.metagraph.nodes:
                hvs = g.ndata[metagraph_node]['hv']
                embeddings.append((self.node_gatings[metagraph_node](hvs) * self.node_to_graphs[metagraph_node](hvs)).sum(0, keepdim=True))
            return torch.cat(embeddings, dim=0).sum(dim=0, keepdim=True)


# noinspection DuplicatedCode
class GraphProp(nn.Module):
    def __init__(self, num_prop_rounds, node_hidden_size):
        super(GraphProp, self).__init__()

        self.num_prop_rounds = num_prop_rounds

        # Setting from the paper
        self.node_activation_hidden_size = 2 * node_hidden_size

        message_funcs = []
        self.reduce_funcs = []
        node_update_funcs = []

        for t in range(num_prop_rounds):
            # input being [vector hv, vector hu, scalar xuv]
            # scalar xuv is meant to be the edge attribute
            # I'm going to expand it into a vector xuv
            message_funcs.append(nn.Linear(2 * node_hidden_size + 1,
                                           self.node_activation_hidden_size))

            self.reduce_funcs.append(partial(self.dgmg_reduce, round=t))
            node_update_funcs.append(
                nn.GRUCell(self.node_activation_hidden_size,
                           node_hidden_size))

        self.message_funcs = nn.ModuleList(message_funcs)
        self.node_update_funcs = nn.ModuleList(node_update_funcs)

    def dgmg_msg(self, edges):
        """For an edge u->v, return concat([hu, xuv])"""
        return {'m': torch.cat([edges.src['hv'],
                                edges.data['he']],
                               dim=1)}

    def dgmg_reduce(self, nodes, round):
        hv_old = nodes.data['hv']
        m = nodes.mailbox['m']
        message = torch.cat([
            hv_old.unsqueeze(1).expand(-1, m.size(1), -1), m], dim=2)
        node_activation = (self.message_funcs[round](message)).sum(1)

        return {'a': node_activation}

    def forward(self, g):
        if g.number_of_edges() == 0:
            return
        else:
            for t in range(self.num_prop_rounds):
                g.update_all(message_func=self.dgmg_msg, reduce_func=self.reduce_funcs[t])
                g.ndata['hv'] = self.node_update_funcs[t](g.ndata['a'], g.ndata['hv'])


def bernoulli_action_log_prob(logit, action):
    """Calculate the log p of an action with respect to a Bernoulli
    distribution. Use logit rather than prob for numerical stability."""
    if action == 0:
        return F.logsigmoid(-logit)
    else:
        return F.logsigmoid(logit)


def multinomial_action_log_prob(logits, action):
    """Calculate the log p of an action with respect to a Multinomial probability
    distribution. Use logits rather than probs for numerical stability."""
    return F.log_softmax(logits, dim=-1)[:, action: action + 1]


class AddNode(nn.Module):
    def __init__(self, graph_embed_func, node_hidden_size, metagraph):
        super(AddNode, self).__init__()

        self.graph_op = {'embed': graph_embed_func}
        self.metagraph = metagraph

        self.stop = 0
        # Input is the graph readout vector
        # Output is a vector with cardinality of node types + 1 (expressing the no-node/terminating option)
        self.add_node = nn.Linear(graph_embed_func.graph_hidden_size, len(self.metagraph.nodes) + 1)

        # If to add a node, initialize its hv
        self.node_type_embed = nn.Embedding(len(self.metagraph.nodes), node_hidden_size)
        self.initialize_hv = nn.Linear(node_hidden_size + \
                                       graph_embed_func.graph_hidden_size,
                                       node_hidden_size)

        self.init_node_activation = torch.zeros(1, 2 * node_hidden_size)

    def _initialize_node_repr(self, g, node_type, graph_embed): # node_type is an index
        num_nodes = g[self.metagraph.index_to_ntype[node_type]].number_of_nodes()
        hv_init = self.initialize_hv(
            torch.cat([
                self.node_type_embed(torch.LongTensor([node_type])),
                graph_embed], dim=1))
        g[self.metagraph.index_to_ntype[node_type]].nodes[num_nodes - 1].data['hv'] = hv_init
        g[self.metagraph.index_to_ntype[node_type]].nodes[num_nodes - 1].data['a'] = self.init_node_activation

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, action=None):
        graph_embed = self.graph_op['embed'](g)

        logits = self.add_node(graph_embed)
        prob = F.softmax(logits, dim=-1)

        if not self.training:
            action = Categorical(prob).sample().item()
        stop = bool(action == self.stop)

        if not stop:
            g.add_nodes(num=1, ntype=self.metagraph.index_to_ntype[action])
            self._initialize_node_repr(g, action, graph_embed)

        if self.training:
            sample_log_prob = multinomial_action_log_prob(logits, action)
            self.log_prob.append(sample_log_prob)

        return stop


class AddEdge(nn.Module):
    def __init__(self, graph_embed_func, node_hidden_size):
        super(AddEdge, self).__init__()

        self.graph_op = {'embed': graph_embed_func}
        self.add_edge = nn.Linear(graph_embed_func.graph_hidden_size + \
                                  node_hidden_size, 1)

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, action=None):
        graph_embed = self.graph_op['embed'](g)
        src_embed = g.nodes[g.number_of_nodes() - 1].data['hv']

        logit = self.add_edge(torch.cat(
            [graph_embed, src_embed], dim=1))
        prob = torch.sigmoid(logit)

        if not self.training:
            action = Bernoulli(prob).sample().item()
        to_add_edge = bool(action == 0)

        if self.training:
            sample_log_prob = bernoulli_action_log_prob(logit, action)
            self.log_prob.append(sample_log_prob)

        return to_add_edge


class ChooseDestAndUpdate(nn.Module):
    def __init__(self, graph_prop_func, node_hidden_size):
        super(ChooseDestAndUpdate, self).__init__()

        self.graph_op = {'prop': graph_prop_func}
        self.choose_dest = nn.Linear(2 * node_hidden_size, 1)

    def _initialize_edge_repr(self, g, src_list, dest_list):
        # For untyped edges, we only add 1 to indicate its existence.
        # For multiple edge types, we can use a one hot representation
        # or an embedding module.
        edge_repr = torch.ones(len(src_list), 1)
        g.edges[src_list, dest_list].data['he'] = edge_repr

    def prepare_training(self):
        self.log_prob = []

    def forward(self, g, dest):
        src = g.number_of_nodes() - 1
        possible_dests = range(src)

        src_embed_expand = g.nodes[src].data['hv'].expand(src, -1)
        possible_dests_embed = g.nodes[possible_dests].data['hv']   # talan itt lehetne majd a jovoben szurni a halmazon (pl. regi esemenyeket kivenni)
        # tovabba itt kellene szurni, h iranyitott graf eseten csak olyat amibe mutat el, de azt lehet nem itt kell, hanem reprezentaciofrissiteskor
        dests_scores = self.choose_dest(
            torch.cat([possible_dests_embed,
                       src_embed_expand], dim=1)).view(1, -1) # ez a view a vegen csak egy transpose
        dests_probs = F.softmax(dests_scores, dim=1)

        if not self.training:
            dest = Categorical(dests_probs).sample().item()

        if not g.has_edge_between(src, dest): # ez a feltetel valoszinuleg nem kell
            # For undirected graphs, we add edges for both directions
            # so that we can perform graph propagation.
            src_list = [src, dest]
            dest_list = [dest, src] # nekunk eleg lesz csak ez az egy irany majd

            g.add_edges(src_list, dest_list)
            self._initialize_edge_repr(g, src_list, dest_list)

            self.graph_op['prop'](g) # valszeg itt tortenik a reprezentaciofrissites

        if self.training:
            if dests_probs.nelement() > 1:
                self.log_prob.append(
                    F.log_softmax(dests_scores, dim=1)[:, dest: dest + 1])


class DGMG(nn.Module):
    def __init__(self, v_max, node_hidden_size,
                 num_prop_rounds, canonical_etypes):
        super(DGMG, self).__init__()

        # Graph configuration
        self.v_max = v_max # Maximum number of nodes in the graph
        self.metagraph = Metagraph(canonical_etypes=canonical_etypes)

        # Graph embedding module
        self.graph_embed = GraphEmbed(node_hidden_size, self.metagraph)

        # Graph propagation module
        self.graph_prop = GraphProp(num_prop_rounds,
                                    node_hidden_size)

        # Actions
        self.add_node_agent = AddNode(
            self.graph_embed, node_hidden_size, metagraph=self.metagraph)
        self.add_edge_agent = AddEdge(
            self.graph_embed, node_hidden_size)
        self.choose_dest_agent = ChooseDestAndUpdate(
            self.graph_prop, node_hidden_size)

        # Weight initialization
        self.init_weights()

    def init_weights(self):
        from utils import weights_init, dgmg_message_weight_init

        self.graph_embed.apply(weights_init)
        self.graph_prop.apply(weights_init)
        self.add_node_agent.apply(weights_init)
        self.add_edge_agent.apply(weights_init)
        self.choose_dest_agent.apply(weights_init)

        self.graph_prop.message_funcs.apply(dgmg_message_weight_init)

    @property
    def action_step(self):
        old_step_count = self.step_count
        self.step_count += 1

        return old_step_count

    def prepare_for_train(self):
        self.step_count = 0

        self.add_node_agent.prepare_training()
        self.add_edge_agent.prepare_training()
        self.choose_dest_agent.prepare_training()

    def add_node_and_update(self, a=None):
        """Decide if to add a new node.
        If a new node should be added, update the graph."""

        return self.add_node_agent(self.g, a)

    def add_edge_or_not(self, a=None):
        """Decide if a new edge should be added."""

        return self.add_edge_agent(self.g, a)

    def choose_dest_and_update(self, a=None):
        """Choose destination and connect it to the latest node.
        Add edges for both directions and update the graph."""

        self.choose_dest_agent(self.g, a)

    def get_log_prob(self):
        return torch.cat(self.add_node_agent.log_prob).sum()\
               + torch.cat(self.add_edge_agent.log_prob).sum()\
               + torch.cat(self.choose_dest_agent.log_prob).sum()

    def forward_train(self, actions):
        self.prepare_for_train()

        stop = self.add_node_and_update(a=actions[self.action_step])

        while not stop:
            to_add_edge = self.add_edge_or_not(a=actions[self.action_step])
            while to_add_edge:
                self.choose_dest_and_update(a=actions[self.action_step])
                to_add_edge = self.add_edge_or_not(a=actions[self.action_step])
            stop = self.add_node_and_update(a=actions[self.action_step])

        return self.get_log_prob()

    def forward_inference(self):
        stop = self.add_node_and_update()
        while (not stop) and (self.g.number_of_nodes() < self.v_max + 1):
            num_trials = 0
            to_add_edge = self.add_edge_or_not()
            while to_add_edge and (num_trials < self.g.number_of_nodes() - 1):
                self.choose_dest_and_update()
                num_trials += 1
                to_add_edge = self.add_edge_or_not()
            stop = self.add_node_and_update()

        return self.g

    def forward(self, actions=None):
        # The graph we will work on
        self.g = copy.deepcopy(self.metagraph.empty_graph)

        # If there are some features for nodes and edges,
        # zero tensors will be set for those of new nodes and edges.
        for ntype in self.g.ntypes:
            self.g.set_n_initializer(dgl.frame.zero_initializer, ntype=ntype)
        for etype in self.g.canonical_etypes:
            self.g.set_e_initializer(dgl.frame.zero_initializer, etype=etype)

        if self.training:
            return self.forward_train(actions)
        else:
            return self.forward_inference()
