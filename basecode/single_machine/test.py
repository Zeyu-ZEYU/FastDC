import torch
from softmax import softmax

score = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float16, device="cuda")
score = score.view(1, 1, 1, -1)
torch_softmax = torch.nn.Softmax(dim=3)

from megatron.model.enums import AttnMaskType
from megatron.model.fused_softmax import FusedScaleMaskSoftmax


def attention_mask_func(attention_scores, attention_mask):
    attention_scores.masked_fill_(attention_mask, -10000.0)
    return attention_scores


scale_mask_softmax = FusedScaleMaskSoftmax(
    True,
    False,
    AttnMaskType.padding,
    True,
    attention_mask_func,
    True,
    24,
)

# mask = torch.ones([20, 20]).view(1, 1, 20, 20) < 0.5
# mask = mask.to("cuda")
p1 = softmax(score, 1, False)


sm = torch.nn.Softmax(dim=3)
# p1 = sm(score)
print(p1)


# print(torch.max(torch.abs(p1 - p2)))
