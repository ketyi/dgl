import dgl
import torch

g = dgl.heterograph({
    ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    ('user', 'anything', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
    ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
})

print(g.etypes)
print(g.num_edges)
print(g.in_degrees(etype=('user', 'follows', 'user')))
print(g.out_degrees(etype=('user', 'follows', 'user')))