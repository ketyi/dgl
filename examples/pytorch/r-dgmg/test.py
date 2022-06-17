import dgl
import torch as th
import dgl.nn.pytorch as dglnn
import sys

from numpy.distutils.system_info import numpy_info

'''
g = dgl.heterograph({('A', 'AB', 'B'): ([], [])}, num_nodes_dict={'A': 0, 'B': 0})
print(g.num_edges)
print(g.num_nodes)
print(g.ntypes)
print(g.etypes)
print(g.canonical_etypes)

g.add_nodes(num=1, ntype='B')
print(g.num_edges)
print(g.num_nodes)
print(g.ntypes)
print(g.etypes)
print(g.canonical_etypes)
'''

graph_data = {
    ('event', 'directly_follows', 'event'): (th.tensor([0, 0, 0, 1, 1, 2, 3, 4]), th.tensor([5, 5, 2, 2, 3, 4, 4, 5]))
}
g = dgl.heterograph(graph_data)



graph_data_2 = {
    ('event', 'directly_follows', 'event'): (th.tensor([0]), th.tensor([1])),
    ('event', 'correlated', 'entity_type_0'): (th.tensor([0]), th.tensor([0]))
}

g_2 = dgl.heterograph(graph_data_2)
print(g_2.metagraph())
sys.exit(0)

print(g_2.num_edges)
g_2.add_nodes(num=1, ntype='event')
print(g_2.num_edges)
#print(g_2.ntypes)
#print(g_2.etypes)
#print(g_2.canonical_etypes)
g_2.add_edges(u=0, v=1, etype='directly_follows')
print(g_2.num_edges)
g_2.add_edges(u=0, v=9, etype='correlated')
print(g_2.num_edges)
sys.exit(0)

# g.nodes['event'].data['hv'] = th.ones(4, 1)
# print(g.nodes['event'].data['hv'])

# g.edges['directly_follows'].data['he'] = th.zeros(3, 1)
# print(g.edges['directly_follows'].data['he'])

hetero_graph_conv = dglnn.HeteroGraphConv({'directly_follows' : dglnn.GraphConv(1, 2, norm='none')}, aggregate='sum')
event_features = {'event' : th.ones((g.number_of_nodes('event'), 1))}

#features = th.ones(4, 8)
#conv = dglnn.conv.GraphConv(8, 2, allow_zero_in_degree=True)
#res = conv(g, features)
#print(res.shape)

res_2 = hetero_graph_conv(g, event_features)
print(res_2['event'].shape)


g_3 = dgl.graph(([0, 0, 0, 1, 1, 2, 3, 4], [5, 5, 2, 2, 3, 4, 4, 5]))
feat_3 = th.ones(6, 1)
conv_3 = dglnn.RelGraphConv(1, 1, 1)
etype_3 = th.tensor([0, 0, 0, 0, 0, 0, 0, 0])
res_3 = conv_3(g_3, feat_3, etype_3, norm=None)
print(res_3)