from torch.utils.data import Dataset


# An example graph
class HeteroGraphDataset(Dataset):
    def __init__(self):
        """Generate a hetero graph according to a sequence of actions.

        Parameters
        ----------
        actions : list of 2-tuples of int
            actions[t] gives (i, j), the action to execute by DGMG at timestep t.
            - If i = 0, j specifies either the type of the node to add or termination (N-th)
            - If i = 1, j specifies activity attribute of the event node if an event node was added in the previous step
            - If i = 2, j specifies either the type of the edge to add or termination (E-th)
            - If i = 3, j specifies the destination node id for the new edge to add
            - If i = 4, j specifies the direction for the edge to add:
                - If j = 0: incoming (in the viewpoint of the newly added node)
                - If j = 1: outgoing (in the viewpoint of the newly added node)
            We model the random variables of 2, 3, 4 jointly --> joint i = 6
            - If i = 5, j specifies the entity type of correlation attribute if a directly follows edge was added in the previous step
        """
        super().__init__()
        self.dataset = [(0, 0), (1, 0), (6, 4), (0, 1), (6, 2), (6, 8), (0, 2), (6, 1), (6, 12), (0, 0), (1, 1), (6, 4), (6, 8), (6, 3), (5, 0), (6, 3), (5, 1), (6, 16), (0, 0), (1, 2), (6, 20), (0, 3), (6, 18), (6, 24), (0, 0), (1, 3), (6, 20), (6, 19), (5, 2), (6, 28), (0, 0), (1, 4), (6, 20), (6, 8), (6, 27), (5, 2), (6, 15), (5, 1), (6, 32), (0, 4)]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def collate_single(self, batch):
        assert len(batch) == 1, 'Currently we do not support batched training'
        return batch[0]

    def collate_batch(self, batch):
        return batch