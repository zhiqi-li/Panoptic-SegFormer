import torch
from torch import nn
from operator import itemgetter

def exists(val):
    return val is not None


def map_el_ind(arr, ind):
    return list(map(itemgetter(ind), arr))


def sort_and_return_indices(arr):
    indices = [ind for ind in range(len(arr))]
    arr = zip(arr, indices)
    arr = sorted(arr)
    return map_el_ind(arr, 0), map_el_ind(arr, 1)


def calculate_permutations(num_dimensions, emb_dim):
    total_dimensions = num_dimensions + 2
    emb_dim = emb_dim if emb_dim > 0 else (emb_dim + total_dimensions)
    xy_dims = [ind for ind in range(1, total_dimensions) if ind != emb_dim]

    permutations = []

    for xy_dim in xy_dims:
        last_two_dims = [xy_dim, emb_dim]
        dims_rest = set(range(0, total_dimensions)) - set(last_two_dims)
        permutation = [*dims_rest, *last_two_dims]
        permutations.append(permutation)

    return permutations


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.tensor(0.))

    def forward(self, x):
        return self.fn(x) * self.g


class Sequential(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = blocks

    def forward(self, x):
        for f, g in self.blocks:
            x = x + f(x) + g(x)
        return x


class PermuteToFrom(nn.Module):
    def __init__(self, permutation, fn):
        super().__init__()
        self.fn = fn
        _, inv_permutation = sort_and_return_indices(permutation)
        self.permutation = permutation
        self.inv_permutation = inv_permutation

    def forward(self, x, **kwargs):
        xy = x.permute(*self.permutation).contiguous()

        shape = xy.shape
        *_, t, d = shape

        xy = xy.reshape(-1, t, d)

        xy = self.fn(xy, **kwargs)

        xy = xy.reshape(*shape)
        xy = xy.permute(*self.inv_permutation).contiguous()
        return xy


class PositionalEmbedding(nn.Module):
    def __init__(self, dim, shape, emb_dim_index=1):
        super().__init__()
        parameters = []
        total_dimensions = len(shape) + 2
        ax_dim_indexes = [i for i in range(1, total_dimensions) if i != emb_dim_index]

        for xy_dim, xy_dim_index in zip(shape, ax_dim_indexes):
            shape = [1] * total_dimensions
            shape[emb_dim_index] = dim
            shape[xy_dim_index] = xy_dim
            parameter = nn.Parameter(torch.randn(*shape))
            parameters.append(parameter)

        self.params = nn.ParameterList(parameters)

    def forward(self, x):
        for param in self.params:
            x = x + param
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_heads=None):
        super().__init__()
        self.dim_heads = (dim // heads) if dim_heads is None else dim_heads
        dim_hidden = self.dim_heads * heads

        self.heads = heads
        self.to_q = nn.Linear(dim, dim_hidden, bias=False)
        self.to_kv = nn.Linear(dim, 2 * dim_hidden, bias=False)
        self.to_out = nn.Linear(dim_hidden, dim)

    def forward(self, x, kv=None):
        kv = x if kv is None else kv
        q, k, v = (self.to_q(x), *self.to_kv(kv).chunk(2, dim=-1))

        b, t, d, h, e = *q.shape, self.heads, self.dim_heads

        merge_heads = lambda x: x.reshape(b, -1, h, e).transpose(1, 2).reshape(b * h, -1, e)
        q, k, v = map(merge_heads, (q, k, v))

        dots = torch.einsum('bie,bje->bij', q, k) * (e ** -0.5)
        dots = dots.softmax(dim=-1)
        out = torch.einsum('bij,bje->bie', dots, v)

        out = out.reshape(b, h, -1, e).transpose(1, 2).reshape(b, -1, d)
        out = self.to_out(out)
        return out


class TwinTransformer(nn.Module):
    def __init__(self, dim=256, depth=12, heads=8, dim_heads=None, dim_index=1, reversible=False, xy_pos_emb_shape=None):
        super().__init__()
        permutations = calculate_permutations(2, dim_index)

        get_ff = lambda: nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1)
        )

        self.pos_emb = PositionalEmbedding(dim, xy_pos_emb_shape, dim_index) if exists(
            xy_pos_emb_shape) else nn.Identity()

        layers = nn.ModuleList([])
        for _ in range(depth):
            attn_functions = nn.ModuleList(
                [PermuteToFrom(permutation, Rezero(SelfAttention(dim, heads, dim_heads))) for permutation in
                 permutations])
            conv_functions = nn.ModuleList([Rezero(get_ff()), Rezero(get_ff())])
            layers.append(attn_functions)
            layers.append(conv_functions)

        execute_type = Sequential
        self.layers = execute_type(layers)

    def forward(self, x):
        x = self.pos_emb(x)
        x = torch.cat((x, x), dim=-1)
        x = self.layers(x)
        return torch.stack(x.chunk(2, dim=-1)).mean(dim=0)