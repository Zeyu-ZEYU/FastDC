import torch

from fastdc.kernels.layout import layer_layout
from fastdc.kernels.reference import gather_tiles, paged_attention, write_kv


def test_reference_paged_attention_matches_dense():
    torch.manual_seed(0)
    hkv, g, d, bs = 2, 3, 16, 4
    kd, vd = [5, 9], [7, 3]
    lay = layer_layout(kd, vd, bs)
    pages = torch.zeros(8, lay.page_elems)
    lens = [6, 5]
    table = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    keys = [torch.randn(n, hkv, d) for n in lens]
    vals = [torch.randn(n, hkv, d) for n in lens]
    for h in range(hkv):
        for b in range(2):
            keys[b][:, h, kd[h]:] = 0
            vals[b][:, h, vd[h]:] = 0

    def slots(b, n):
        return torch.tensor([int(table[b, p // bs]) * bs + p % bs for p in range(n)])

    for b in range(2):
        write_kv(pages, lay, keys[b], vals[b], slots(b, lens[b]))
    qlens = [6, 2]
    q = torch.randn(sum(qlens), hkv * g, d)
    for h in range(hkv):
        q[:, h * g:(h + 1) * g, kd[h]:] = 0
    out = torch.empty(8, hkv * g, d)
    paged_attention(q, pages, lay.k, pages, lay.v, table, torch.tensor([0, 6, 8], dtype=torch.int32),
                    torch.tensor(lens, dtype=torch.int32), 0.25, out)
    ref = torch.zeros_like(out)
    for b, (q0, n) in enumerate(((0, 6), (6, 2))):
        length = lens[b]
        for hq in range(hkv * g):
            h = hq // g
            s = q[q0:q0 + n, hq] @ keys[b][:, h].T * 0.25
            pos = torch.arange(length - n, length)
            s = s.masked_fill(torch.arange(length).unsqueeze(0) > pos.unsqueeze(1), float("-inf"))
            ref[q0:q0 + n, hq, :vd[h]] = (torch.softmax(s, -1) @ vals[b][:, h])[:, :vd[h]]
    assert torch.allclose(out, ref, atol=1e-5)
    assert torch.equal(gather_tiles(pages, lay.k, table[1], 5, max(kd)), keys[1].transpose(0, 1)[:, :, :max(kd)])
