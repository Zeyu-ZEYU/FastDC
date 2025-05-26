import pickle

import einops
import torch

data_path = "/home/zeyu/data/zeroc/"


# qkv = None
# o_proj = None

# with open(f"{data_path}/qkv_iter_0_29.pkl", "rb") as f:
#     qkv = pickle.load(f)

# # with open(f"{data_path}/o_proj.pkl", "rb") as f:
# #     o_proj = pickle.load(f)


# qkv_batch_0 = qkv[10]
# qkv_batch_1 = qkv[11]
# qkv_batch_2 = qkv[12]
# qkv_batch_3 = qkv[13]
# qkv_batch_4 = qkv[14]
# qkv_batch_5 = qkv[15]
# qkv_batch_6 = qkv[16]
# qkv_batch_7 = qkv[17]
# qkv_batch_8 = qkv[18]
# qkv_batch_9 = qkv[19]


# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """
#     This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
#     num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
#     """
#     batch, num_key_value_heads, slen, head_dim = hidden_states.shape
#     if n_rep == 1:
#         return hidden_states
#     hidden_states = hidden_states[:, :, None, :, :].expand(
#         batch, num_key_value_heads, n_rep, slen, head_dim
#     )
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# qk_layers = []
# for b0l, b1l, b2l, b3l, b4l, b5l, b6l, b7l, b8l, b9l in zip(
#     qkv_batch_0,
#     qkv_batch_1,
#     qkv_batch_2,
#     qkv_batch_3,
#     qkv_batch_4,
#     qkv_batch_5,
#     qkv_batch_6,
#     qkv_batch_7,
#     qkv_batch_8,
#     qkv_batch_9,
# ):
#     q0, k0, _ = b0l
#     q1, k1, _ = b1l
#     q2, k2, _ = b2l
#     q3, k3, _ = b3l
#     q4, k4, _ = b4l
#     q5, k5, _ = b5l
#     q6, k6, _ = b6l
#     q7, k7, _ = b7l
#     q8, k8, _ = b8l
#     q9, k9, _ = b9l
#     qk_layers.append(
#         einops.rearrange(
#             torch.cat(
#                 [
#                     q0,
#                     q1,
#                     q2,
#                     q3,
#                     q4,
#                     q5,
#                     q6,
#                     q7,
#                     q8,
#                     q9,
#                     repeat_kv(k0, 4),
#                     repeat_kv(k1, 4),
#                     repeat_kv(k2, 4),
#                     repeat_kv(k3, 4),
#                     repeat_kv(k4, 4),
#                     repeat_kv(k5, 4),
#                     repeat_kv(k6, 4),
#                     repeat_kv(k7, 4),
#                     repeat_kv(k8, 4),
#                     repeat_kv(k9, 4),
#                 ],
#                 dim=2,
#             ),
#             "b h t d -> h (b t) d",
#         )
#     )


# with open(f"{data_path}/tmp.pkl", "wb") as f:
#     pickle.dump(qk_layers, f)

with open(f"{data_path}/tmp.pkl", "rb") as f:
    qk_layers = pickle.load(f)

with open(f"{data_path}/qk_rotation.pkl", "rb") as f:
    rot_layers = pickle.load(f)

sums = []
for qk, r in zip(qk_layers, rot_layers):
    _, _, vh = torch.linalg.svd(qk.to(torch.float), False)
    v = vh.transpose(1, 2).to(torch.float16)
    # diff = v - r
    # sumall = torch.sqrt(torch.sum(diff**2)) / diff.numel()
    # sums.append(sumall)
    a = torch.sum(torch.sqrt(v**2)) / v.numel()
    sums.append(a)

print(sums)
