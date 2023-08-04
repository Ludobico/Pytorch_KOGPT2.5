import torch
from xformers.components.attention import ScaledDotProduct

device = 'cuda' if torch.cuda.is_available() else 'cpu'

attention = ScaledDotProduct().cuda()

inputs = torch.rand((16, 1024, 1024), device=device)

# 마스킹 분리전 cuda 메모리 초기화
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

mask = (torch.rand((1024, 1024)) < 0.9).cuda()
att = attention(q=inputs, k=inputs, v=inputs, mask=mask)

torch.cuda.synchronize()
max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
print(f"Dense - Peak memory use: {max_memory}MB")

# 메모리 변경 확인을 위해 다시한번 초기화
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

mask = (torch.rand((1024, 1024)) < 0.1).cuda()
att = attention(q=inputs, k=inputs, v=inputs, mask=mask)

torch.cuda.synchronize()
max_memory = torch.cuda.max_memory_allocated() // 2 ** 20
print(f"Sparse - Peak memory use: {max_memory}MB")
