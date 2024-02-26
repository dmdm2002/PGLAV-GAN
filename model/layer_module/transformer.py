import torch
import torch.nn as nn

from model.layer_module.attention import ParallelAttentionModule, SequenceAttentionModule


class PositionWiseFFN(nn.Module):
    def __init__(self, in_features, ffn_features) -> None:
        super().__init__()

        self.FFN = nn.Sequential(
            nn.Linear(in_features, ffn_features),
            nn.GELU(),
            nn.Linear(ffn_features, in_features),
        )

    def forward(self, x):
        return self.FFN(x)


class TransBlock(nn.Module):
    def __init__(self, in_features, out_features, rezero=True):
        super(TransBlock, self).__init__()

        self.norm1 = nn.LayerNorm(in_features)

        # self.parallel_attn = ParallelAttentionModule(in_features, num_heads=8, window_size=7)
        self.sequence_attn = SequenceAttentionModule(in_features, num_heads=8, window_size=7)

        self.norm2 = nn.LayerNorm(in_features)
        self.ffn = PositionWiseFFN(in_features, out_features)

        self.rezero = rezero

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1,)))
        else:
            self.re_alpha = 1

    def forward(self, x):
        y1 = self.norm1(x)

        y1 = self.sequence_attn(y1, self.re_alpha)

        y = x + self.re_alpha * y1

        y2 = self.norm2(y)
        y2 = self.ffn(y2)

        y = y + self.re_alpha * y2

        return y


class FourierEmbedding(nn.Module):
    # arXiv: 2011.13775

    def __init__(self, features, height, width):
        super().__init__()
        self.projector = nn.Linear(2, features)
        self._height = height
        self._width = width

    def forward(self, y, x):
        # x : (N, L)
        # y : (N, L)
        x_norm = 2 * x / (self._width - 1) - 1
        y_norm = 2 * y / (self._height - 1) - 1

        # z : (N, L, 2)
        z = torch.cat((x_norm.unsqueeze(2), y_norm.unsqueeze(2)), dim=2)

        return torch.sin(self.projector(z))


class TransformerEncoder(nn.Module):

    def __init__(
            self, features, ffn_features, n_blocks, rezero=True):
        super().__init__()

        self.encoder = nn.Sequential(*[
            TransBlock(
                features, ffn_features, rezero,
            ) for _ in range(n_blocks)
        ])

    def forward(self, x):
        # x : (N, L, features)

        # y : (L, N, features)
        y = x.permute((1, 0, 2))
        y = self.encoder(y)

        # result : (N, L, features)
        result = y.permute((1, 0, 2))

        return result


class ViTInput(nn.Module):
    def __init__(
            self, input_features, embed_features, features, height, width):
        super().__init__()
        self._height = height
        self._width = width

        x = torch.arange(width).to(torch.float32)
        y = torch.arange(height).to(torch.float32)

        x, y = torch.meshgrid(x, y)
        self.x = x.reshape((1, -1))
        self.y = y.reshape((1, -1))

        self.register_buffer('x_const', self.x)
        self.register_buffer('y_const', self.y)

        self.embed = FourierEmbedding(embed_features, height, width)
        self.output = nn.Linear(embed_features + input_features, features)

    def forward(self, x):
        # x     : (N, L, input_features)
        # embed : (1, height * width, embed_features)
        #       = (1, L, embed_features)
        embed = self.embed(self.y_const, self.x_const)

        # embed : (1, L, embed_features)
        #      -> (N, L, embed_features)
        embed = embed.expand((x.shape[0], *embed.shape[1:]))

        # result : (N, L, embed_features + input_features)
        result = torch.cat([embed, x], dim=2)

        # (N, L, features)
        return self.output(result)


class PixelwiseViT(nn.Module):
    def __init__(
            self, features, n_blocks, ffn_features, embed_features, image_shape, rezero=True):
        super().__init__()

        self.image_shape = image_shape

        self.trans_input = ViTInput(
            image_shape[0], embed_features, features,
            image_shape[1], image_shape[2],
        )

        self.encoder = TransformerEncoder(
            features, ffn_features, n_blocks, rezero
        )

        # self.cam = CAM(features, 16)

        self.trans_output = nn.Linear(features, image_shape[0])

    def forward(self, x):
        # x : (N, C, H, W)

        # itokens : (N, C, H * W)
        itokens = x.view(*x.shape[:2], -1)

        # itokens : (N, C,     H * W)
        #        -> (N, H * W, C    )
        #         = (N, L,     C)
        itokens = itokens.permute((0, 2, 1))

        # y : (N, L, features)
        y = self.trans_input(itokens)
        y = self.encoder(y)

        # otokens : (N, L, C)
        otokens = self.trans_output(y)

        # otokens : (N, L, C)
        #        -> (N, C, L)
        #         = (N, C, H * W)
        otokens = otokens.permute((0, 2, 1))

        # result : (N, C, H, W)
        result = otokens.view(*otokens.shape[:2], *self.image_shape[1:])

        # skip connection and channel attention
        # cam_x = self.cam(result)
        # outpout = x + cam_x

        return result