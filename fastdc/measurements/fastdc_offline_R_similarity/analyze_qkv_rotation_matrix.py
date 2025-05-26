import pickle

import torch

data_path = "/home/zeyu/data/zeroc/"


# qkv = None
# o_proj = None

# with open(f"{data_path}/qkv_iter_0_29.pkl", "rb") as f:
#     qkv = pickle.load(f)

# with open(f"{data_path}/o_proj.pkl", "rb") as f:
#     o_proj = pickle.load(f)


# qkv_batch_0 = qkv[0]
# qkv_batch_1 = qkv[1]
# qkv_batch_2 = qkv[2]
# qkv_batch_3 = qkv[3]
# qkv_batch_4 = qkv[4]
# qkv_batch_5 = qkv[5]
# qkv_batch_6 = qkv[6]
# qkv_batch_7 = qkv[7]
# qkv_batch_8 = qkv[8]
# qkv_batch_9 = qkv[9]

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
#         [q0, q1, q2, q3, q4, q5, q6, q7, q8, q9, k0, k1, k2, k3, k4, k5, k6, k7, k8, k9]
#     )


# with open(f"{data_path}/tmp.pkl", "wb") as f:
#     pickle.dump(qk_layers, f)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# qk_layers = []
# with open(f"{data_path}/tmp.pkl", "rb") as f:
#     qkv_layers = pickle.load(f)
#     for (
#         q0,
#         q1,
#         q2,
#         q3,
#         q4,
#         q5,
#         q6,
#         q7,
#         q8,
#         q9,
#         k0,
#         k1,
#         k2,
#         k3,
#         k4,
#         k5,
#         k6,
#         k7,
#         k8,
#         k9,
#     ) in qkv_layers:
#         k0 = repeat_kv(k0, 4)
#         k1 = repeat_kv(k1, 4)
#         k2 = repeat_kv(k2, 4)
#         k3 = repeat_kv(k3, 4)
#         k4 = repeat_kv(k4, 4)
#         k5 = repeat_kv(k5, 4)
#         k6 = repeat_kv(k6, 4)
#         k7 = repeat_kv(k7, 4)
#         k8 = repeat_kv(k8, 4)
#         k9 = repeat_kv(k9, 4)
#         qk_layers.append(
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
#                     k0,
#                     k1,
#                     k2,
#                     k3,
#                     k4,
#                     k5,
#                     k6,
#                     k7,
#                     k8,
#                     k9,
#                 ],
#                 dim=2,
#             )
#         )


# with open(f"{data_path}/qk.pkl", "wb") as f:
#     pickle.dump(qk_layers, f)


# with open(f"{data_path}/qk.pkl", "rb") as f:
#     qk_layers = pickle.load(f)


# import einops

# arr = []
# for tsr in qk_layers:
#     newtsr = einops.rearrange(tsr, "a b c d -> b (a c) d")
#     _, _, vh = torch.linalg.svd(newtsr.to("cuda:0").float(), False)
#     v = vh.transpose(1, 2).to(torch.float16).cpu()
#     arr.append(v)

# with open(f"{data_path}/qk_rotation.pkl", "wb") as f:
#     pickle.dump(arr, f)


# with open(f"{data_path}/qk_rotation.pkl", "rb") as f:
#     layers = pickle.load(f)
# print(layers[0].shape)

# import einops

# with open(f"{data_path}/o_proj.pkl", "rb") as f:
#     layers = pickle.load(f)

# with open(f"{data_path}/qkv_iter_0_29.pkl", "rb") as f:
#     qkv = pickle.load(f)


# qkv_batch_0 = qkv[0]
# qkv_batch_1 = qkv[1]
# qkv_batch_2 = qkv[2]
# qkv_batch_3 = qkv[3]
# qkv_batch_4 = qkv[4]
# qkv_batch_5 = qkv[5]
# qkv_batch_6 = qkv[6]
# qkv_batch_7 = qkv[7]
# qkv_batch_8 = qkv[8]
# qkv_batch_9 = qkv[9]

# v_layers = []
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
#     _, _, v0 = b0l
#     _, _, v1 = b1l
#     _, _, v2 = b2l
#     _, _, v3 = b3l
#     _, _, v4 = b4l
#     _, _, v5 = b5l
#     _, _, v6 = b6l
#     _, _, v7 = b7l
#     _, _, v8 = b8l
#     _, _, v9 = b9l
#     v0 = repeat_kv(v0, 4)
#     v1 = repeat_kv(v1, 4)
#     v2 = repeat_kv(v2, 4)
#     v3 = repeat_kv(v3, 4)
#     v4 = repeat_kv(v4, 4)
#     v5 = repeat_kv(v5, 4)
#     v6 = repeat_kv(v6, 4)
#     v7 = repeat_kv(v7, 4)
#     v8 = repeat_kv(v8, 4)
#     v9 = repeat_kv(v9, 4)
#     v_layers.append(
#         einops.rearrange(
#             torch.cat([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9], dim=2),
#             "a b c d -> b (a c) d",
#         )
#     )


# arr = []
# for v, (x, y) in zip(v_layers, layers):
#     tsr = x + y
#     newtsr = einops.rearrange(tsr, "(c d) b -> c b d", c=32, d=128)
#     newtsr = torch.cat([v, newtsr.to(torch.float16)], dim=1)
#     _, _, vh = torch.linalg.svd(newtsr.to("cuda:0").float(), False)
#     v = vh.transpose(1, 2).to(torch.float16).cpu()
#     arr.append(v)

# with open(f"{data_path}/vl_rotation.pkl", "wb") as f:
#     pickle.dump(arr, f)


with open(f"{data_path}/o_proj.pkl", "rb") as f:
    o_proj = pickle.load(f)
