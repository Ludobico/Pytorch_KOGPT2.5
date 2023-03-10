import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class EmbeddingLayer(nn.Module):
    def __init__(self, in_channel, embed_dim, img_size, patch_size):
        super().__init__()
        self.num_tokens = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        self.project = nn.Conv2d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)

        # 문장의 시작부분(BOS)에 삽입되는 토큰
        # nn.parameter 모델의 파라미터를 정의
        self.cls_token = nn.Parameter(torch.zeros(1,1, embed_dim))

        self.num_tokens += 1
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, self.embed_dim))

        nn.init.normal_(self.cls_token, std=1e-6)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        B, C, H, W = x.shape
        embedding = self.project(x)
        z = embedding.view(B, self.embed_dim, -1).permute(0, 2, 1) # BCHW -> BNC

        cls_tokens = self.cls_token.expand(B, -1, -1)
        z = torch.cat([cls_tokens, z], dim=1)

        z = z + self.pos_embed
        return z
    

if __name__ == '__main__':
    img = torch.randn([2, 3, 32, 32])
    embedding = EmbeddingLayer(in_channel=3, embed_dim=192, img_size=32, patch_size=4)
    z = embedding(img)
    print(z.size())
