
# !/usr/bin/env python
import os
from collections import OrderedDict
import clip
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
import torch
from torch import nn
from torch.nn import MultiheadAttention
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups)


def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)


def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)


def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)


def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)


def conv_5x5x5(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)


def bn_3d(dim):
    return nn.BatchNorm3d(dim)

class ConvTransBN(nn.Module):  # (convolution => [BN] => ReLU)
    def __init__(self, in_channels, out_channels):
        super(ConvTransBN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ConvAttention(nn.Module):
    def __init__(self, d_model, dw_reduction=1.5, pos_kernel_size=3):
        super().__init__()

        padding = pos_kernel_size // 2
        re_d_model = int(d_model // dw_reduction)
        self.pos_embed = nn.Sequential(
            nn.BatchNorm3d(d_model),
            nn.Conv3d(d_model, re_d_model, kernel_size=1, stride=1, padding=0),
            nn.Conv3d(re_d_model, re_d_model, kernel_size=(pos_kernel_size, 1, 1), stride=(1, 1, 1),
                      padding=(padding, 0, 0), groups=re_d_model),
            nn.Conv3d(re_d_model, d_model, kernel_size=1, stride=1, padding=0),
        )

        # init zero
        nn.init.constant_(self.pos_embed[3].weight, 0)
        nn.init.constant_(self.pos_embed[3].bias, 0)

    def forward(self, x):
        return self.pos_embed(x)


class AttentionBlock(nn.Module):
    def __init__(
            self, d_model, n_head, attn_mask=None, drop_path=0.0, dw_reduction=1.5
    ):
        super().__init__()

        self.n_head = n_head
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.convatt1 = ConvAttention(d_model, dw_reduction=dw_reduction)
        self.convatt2 = ConvAttention(d_model, dw_reduction=dw_reduction)

        # spatial
        self.attn = MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, T=8):
        # x: 1+HW, NT, C
        # Local MHRA
        tmp_x = x[1:, :, :]
        L, NT, C = tmp_x.shape
        N = NT // T
        H = W = int(L ** 0.5)
        tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
        tmp_x = tmp_x + self.drop_path(self.convatt1(tmp_x))
        tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)
        x = torch.cat([x[:1, :, :], tmp_x], dim=0)

        x = x + self.drop_path(self.attention(self.ln_1(x)))

        tmp_x = x[1:, :, :]
        tmp_x = tmp_x.view(H, W, N, T, C).permute(2, 4, 3, 0, 1).contiguous()
        tmp_x = tmp_x + self.drop_path(self.convatt2(tmp_x))
        tmp_x = tmp_x.view(N, C, T, L).permute(3, 0, 2, 1).contiguous().view(L, NT, C)
        x = torch.cat([x[:1, :, :], tmp_x], dim=0)

        x = x + self.drop_path(self.mlp(self.ln_2(x)))

        return x


class ConvInput(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.num_patches = num_patches

        self.proj = nn.Sequential(
            nn.Conv3d(in_chans, embed_dim // 2, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(embed_dim // 2),
            nn.GELU(),
            nn.Conv3d(embed_dim // 2, embed_dim, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(2, 1, 1)),
            nn.BatchNorm3d(embed_dim),
        )

    def forward(self, x):
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ImageTextBlock(nn.Module):
    def __init__(self, dim, out_dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, slice_num = 3, bert_embedding = True):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)

        if bert_embedding:
            self.conv_text = nn.Conv1d(768, dim, 1, 1)
        else:
            self.conv_text = nn.Conv1d(512, dim, 1, 1)

        self.conv3 = nn.Conv1d(dim * slice_num + dim, dim * slice_num, 1, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, text):
        B, C, T, H, W = x.shape
        x = x + self.pos_embed(x)

        x = torch.cat([self.conv_text(text.permute(0, 2, 1)), x.reshape(B, C*T, H*W)], dim=1)

        x = self.conv3(x).reshape(B, C, T, H, W)

        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))

        x_embedding = self.drop_path(self.mlp(self.norm2(x)))

        return x_embedding


class ImageBlock(nn.Module):
    def __init__(self, dim, out_dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.conv1 = conv_1x1x1(dim, dim, 1)
        self.conv2 = conv_1x1x1(dim, dim, 1)
        self.attn = conv_5x5x5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=out_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, text):
        x = x + self.pos_embed(x)

        x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x)))))

        x_embedding = self.drop_path(self.mlp(self.norm2(x)))

        return x_embedding

class Transformer(nn.Module):
    def __init__(
            self, width, layers, heads, attn_mask=None, backbone_drop_path_rate=0., t_size=8, dw_reduction=2, n_dim=256,
            cls_dropout=0.5, num_classes=2,
    ):
        super().__init__()
        self.T = t_size

        # backbone
        b_dpr = [x.item() for x in torch.linspace(0, backbone_drop_path_rate, layers)]

        self.resblocks = nn.ModuleList([
            AttentionBlock(
                width, heads, attn_mask,
                drop_path=b_dpr[i],
                dw_reduction=dw_reduction,
            ) for i in range(layers)
        ])

        # projection
        self.proj = nn.Sequential(
            nn.LayerNorm(n_dim),
            nn.Dropout(cls_dropout),
            nn.Linear(n_dim, num_classes),
        )

    def forward(self, x):
        T_down = self.T
        L, NT, C = x.shape
        N = NT // T_down

        for i, resblock in enumerate(self.resblocks):
            x = resblock(x, T_down)

        x = x.view(L, N, T_down, C)[0].mean(1)  # L, N, T, C

        return self.proj(x)

class CMB_VisionTransformer(nn.Module):
    def __init__(
            self,
            # backbone
            input_resolution, text_input, patch_size, width, layers, heads, embedding_dim, output_dim, backbone_drop_path_rate=0.,
            t_size=8, n_dim=256, text_ch_in = 20, slice_num = 3,
            cls_dropout=0.5, num_classes=2, bert_embedding = True, clip_image_input = False,
    ):
        super().__init__()
        self.text_input = text_input

        self.input_resolution = input_resolution
        self.output_dim = output_dim

        self.clip_image_input = clip_image_input
        self.bert_embedding = bert_embedding

        self.conv_input = ConvInput(input_resolution, patch_size, 3, embedding_dim)

        if self.bert_embedding and self.clip_image_input:
            self.clip_conv = ConvTransBN(in_channels=512, out_channels = 768)

        if self.text_input:
            if not clip_image_input:
                self.text_conv = ConvTransBN(in_channels=text_ch_in,
                                             out_channels=(input_resolution // patch_size) * (input_resolution // patch_size))
            else:
                self.text_conv = ConvTransBN(in_channels=text_ch_in + slice_num,
                                             out_channels=(input_resolution // patch_size) * (
                                                         input_resolution // patch_size))
            self.embedding_blocks = nn.ModuleList([
                ImageTextBlock(dim=embedding_dim, out_dim=output_dim, mlp_ratio=3, slice_num = slice_num, bert_embedding=bert_embedding)])
        else:
            self.embedding_blocks = nn.ModuleList([
                ImageBlock(dim=embedding_dim, out_dim=output_dim, mlp_ratio=3)])

        t_size = t_size // 2

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(
            width, layers, heads,
            backbone_drop_path_rate=backbone_drop_path_rate, t_size=t_size, n_dim=n_dim,
            cls_dropout=cls_dropout, num_classes=num_classes,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, texts, clip_im):

        x = self.conv_input(x)  # shape = [Batch, Channel, Frames, H, W]

        if self.text_input:
            if not self.clip_image_input:
                texts = self.text_conv(texts)
            else:
                if self.bert_embedding:
                    clip_im = clip_im.permute(0, 2, 1)
                    clip_im = self.clip_conv(clip_im)
                    clip_im = clip_im.permute(0, 2, 1)

                clip_fusion = torch.concat([texts, clip_im], dim=1)
                texts = self.text_conv(clip_fusion)

        for blk in self.embedding_blocks:
            x = blk(x, texts)

        B, C, T, H, W = x.shape

        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        cls_x = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        x = torch.cat([cls_x, x], dim=1)

        x_embed = self.positional_embedding.to(x.dtype)

        x = x + x_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)

        out = self.transformer(x)

        return out


def inflate_weight(weight_2d, time_dim, center=True):

    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 2:
                continue
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim)

    model.load_state_dict(state_dict, strict=False)


def CmbFormer_S(
        input_resolution = 64, text_input = True, patch_size = 4, t_size=6,  backbone_drop_path_rate=0.,
        n_layers=6, n_dim=512, n_head=4,
        cls_dropout=0.5, num_classes=2,
        text_channel_in = 70, slice_num = 3,
        bert_embedding = True, clip_image_input = False
):
    model = CMB_VisionTransformer(
        input_resolution=input_resolution,
        text_input = text_input,
        patch_size=patch_size,
        width=n_dim,
        layers=n_layers,
        heads=n_head,
        embedding_dim = n_dim,
        output_dim=n_dim,
        t_size=t_size,
        backbone_drop_path_rate=backbone_drop_path_rate,
        n_dim=n_dim,
        text_ch_in = text_channel_in,
        slice_num = slice_num,
        cls_dropout=cls_dropout,
        num_classes=num_classes,
        bert_embedding=bert_embedding,
        clip_image_input = clip_image_input
    )

    return model.eval()

def CmbFormer_B(
        input_resolution = 64, text_input = True, patch_size = 4, t_size=6,  backbone_drop_path_rate=0.,
        n_layers=12, n_dim=768, n_head=8,
        cls_dropout=0.5, num_classes=2,
        text_channel_in = 70, slice_num = 3,
        bert_embedding = True, clip_image_input = False
):

    model = CMB_VisionTransformer(
        input_resolution=input_resolution,
        text_input = text_input,
        patch_size=patch_size,
        width=n_dim,
        layers=n_layers,
        heads=n_head,
        embedding_dim = n_dim,
        output_dim=n_dim,
        t_size=t_size,
        backbone_drop_path_rate=backbone_drop_path_rate,
        n_dim=n_dim,
        text_ch_in = text_channel_in,
        slice_num = slice_num,
        cls_dropout=cls_dropout,
        num_classes=num_classes,
        bert_embedding = bert_embedding,
        clip_image_input = clip_image_input
    )

    return model.eval()

