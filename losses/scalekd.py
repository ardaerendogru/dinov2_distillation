import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import math
import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import resize 
from typing import Optional, Dict, Tuple



class ScaleKD(nn.Module):
    def __init__(self,
                 name,
                 use_this,
                 alpha,
                 student_dims,
                 teacher_dims,
                 query_hw,
                 pos_hw,
                 pos_dims,
                 window_shapes=(1,1),
                 self_query=True,
                 softmax_scale=[1,1],
                 dis_freq='high',
                 num_heads=8
                 ):
        super().__init__()
        self.alpha = alpha
        self.dis_freq = dis_freq
        self.self_query = self_query
        self.projector_0 = AttentionProjector(student_dims, teacher_dims, query_hw, pos_dims, window_shapes=window_shapes, self_query=self_query, softmax_scale=softmax_scale[0], num_heads=num_heads)
        self.projector_1 = AttentionProjector(student_dims, teacher_dims, query_hw, pos_dims, window_shapes=window_shapes, self_query=self_query, softmax_scale=softmax_scale[1], num_heads=num_heads)
    def forward(self,
                preds_S: torch.Tensor,
                preds_T: torch.Tensor,
                query_s: Optional[torch.Tensor] = None,
                query_f: Optional[torch.Tensor] = None,
                ) -> Dict[str, torch.Tensor]:
        """Forward function.
        Args:
            preds_S (Tensor): Bs*C*H*W, student's feature map
            preds_T (Tensor): Bs*C*H*W, teacher's feature map
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing losses and similarity.
        """

        preds_S_spat =  self.project_feat_spat(preds_S, query=query_s)
        preds_S_freq =  self.project_feat_freq(preds_S, query=query_f)

        spat_loss, spatial_similarity = self.get_spat_loss(preds_S_spat, preds_T)
        freq_loss, frequency_similarity = self.get_freq_loss(preds_S_freq, preds_T)
        return {'spatial_loss': spat_loss, 
                'frequency_loss': freq_loss, 
                'spatial_similarity': spatial_similarity, 
                'frequency_similarity':frequency_similarity,
                'loss': spat_loss + freq_loss}
    
    def project_feat_spat(self, preds_S, query=None):
        preds_S = self.projector_0(preds_S, query=query)

        return preds_S

    def project_feat_freq(self, preds_S, query=None):
        preds_S = self.projector_1(preds_S, query=query)

        return preds_S


    def get_spat_loss(self, preds_S: torch.Tensor, preds_T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial loss between student and teacher features.
        Args:
            preds_S (Tensor): Student features.
            preds_T (Tensor): Teacher features.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Discrepancy loss and similarity.
        """
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape

        # Project student's features and reshape to (B, teacher_dims, H, W)
        preds_S = preds_S.permute(0, 2, 1).contiguous().view(N, C, H, W)

        # Normalize features for loss computation
        preds_S = F.normalize(preds_S, dim=1)
        preds_T = F.normalize(preds_T, dim=1)

        # Compute MSE loss
        dis_loss_arch_st = loss_mse(preds_S, preds_T) / N
        dis_loss_arch_st = dis_loss_arch_st * self.alpha[0]

        # Compute cosine similarity for monitoring
        similarity = F.cosine_similarity(preds_S, preds_T, dim=1).mean()

        return dis_loss_arch_st, similarity


    def get_freq_loss(self, preds_S: torch.Tensor, preds_T: torch.Tensor) -> torch.Tensor:
        """Compute frequency loss between student and teacher features.
        Args:
            preds_S (Tensor): Student features.
            preds_T (Tensor): Teacher features.
        Returns:
            torch.Tensor: Frequency loss.
        """
        loss_mse = nn.MSELoss(reduction='sum')
        N, C, H, W = preds_T.shape
        device = preds_S.device

        dct = DCT(resolution=H, device=device)

        preds_S = preds_S.permute(0,2,1).contiguous().view(*preds_T.shape)

        preds_S_freq = dct.forward(preds_S)
        preds_T_freq = dct.forward(preds_T)

        preds_S_freq[:,:,0,0]=0
        preds_T_freq[:,:,0,0]=0

        preds_S = dct.inverse(preds_S_freq)
        preds_T = dct.inverse(preds_T_freq)
        preds_S = F.normalize(preds_S, dim=1, p=2)
        preds_T = F.normalize(preds_T, dim=1, p=2)
        
        dis_loss = loss_mse(preds_S, preds_T)/N 

        dis_loss = dis_loss * self.alpha[1]
        similarity = F.cosine_similarity(preds_S, preds_T, dim=1).mean()

        return dis_loss, similarity

    def _compute_affinity_map(self, teacher_features):
        """Compute normalized patch-wise affinity map from teacher features."""
        # Assuming teacher_features is BxCxHxW
        B, C, H, W = teacher_features.shape
        patch_features = teacher_features.flatten(2)  # shape: [B, C, H*W]
        patch_features = F.normalize(patch_features, p=2, dim=1)  # normalize feature vectors along the channel dim

        # Compute cosine similarity between patches:
        corrs = torch.matmul(patch_features.transpose(1, 2), patch_features)  # shape: [B, H*W, H*W]
        corrs = corrs.reshape(B, H, W, H * W).permute(0, 3, 1, 2)  # reshape to expected format: [B, H*W, H, W]
        return corrs

    def compute_weighted_pool(self, maskclip_feats: torch.Tensor, corrs: torch.Tensor) -> torch.Tensor:
        """Weighted pooling method from CLIP-DINOiser paper - REVISED for 3D corrs.
        Args:
            maskclip_feats (torch.Tensor): Raw clip features (student features).
            corrs (torch.Tensor): Correlations as weights (affinity map).
        Returns:
            torch.Tensor: Refined clip features (pooled student features).
        """
        """
        Weighted pooling method.
        :param maskclip_feats: torch.tensor - raw clip features
        :param corrs: torch.tensor - correlations as weights for pooling mechanism
        :return: torch.tensor - refined clip features
        """
        B = maskclip_feats.shape[0]
        h_m, w_m = maskclip_feats.shape[-2:]
        h_w, w_w = corrs.shape[-2:]

        if (h_m != h_w) or (w_m != w_w):
            maskclip_feats = resize(
                input=maskclip_feats,
                size=(h_w, w_w),
                mode='bilinear',
                align_corners=False)
            h_m, w_m = h_w, w_w

        maskclip_feats_ref = torch.einsum("bnij, bcij -> bcn", corrs, maskclip_feats)  # B C HW
        norm_factor = corrs.flatten(-2, -1).sum(dim=-1)[:, None]  # B 1 HW
        maskclip_feats_ref = maskclip_feats_ref / (norm_factor + 1e-6)

        # RESHAPE back to 2d
        maskclip_feats_ref = maskclip_feats_ref.reshape(B, -1, h_m, w_m)
        return maskclip_feats_ref
    


class AttentionProjector(nn.Module):
    def __init__(self,
                 student_dims,
                 teacher_dims,
                 hw_dims,
                 pos_dims,
                 window_shapes=(1,1),
                 self_query=True,
                 softmax_scale=1.,
                 num_heads=8,
                 ):
        super(AttentionProjector, self).__init__()

        self.hw_dims = hw_dims
        self.student_dims = student_dims
        self.teacher_dims = teacher_dims

        self.proj_pos = nn.Sequential(nn.Conv2d(teacher_dims, teacher_dims, 1),
                                      nn.ReLU(),
                                      )

        self.proj_student = nn.Sequential(nn.Conv2d(student_dims, student_dims, 3, stride=1, padding=1),
                                      nn.BatchNorm2d(student_dims),
                                      nn.ReLU())

        self.pos_embed = nn.Parameter(torch.zeros(1, student_dims, hw_dims[0], hw_dims[1]), requires_grad=True)
        
        self.pos_attention = WindowMultiheadPosAttention(teacher_dims, num_heads=num_heads, input_dims=student_dims, pos_dims=pos_dims, window_shapes=window_shapes, softmax_scale=softmax_scale)
        self.ffn = FFN(
            embed_dims=teacher_dims,
            feedforward_channels=teacher_dims * 4,
            num_fcs=2,
            act_cfg=dict(type='ReLU', inplace=True),
            dropout=0.0,
            add_residual=True
        )

        self.norm = nn.LayerNorm([teacher_dims])

        if self_query:
            self.query = nn.Embedding(hw_dims[0] * hw_dims[1], teacher_dims)
        else:
            self.query = None

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)


    def forward(self, x, query=None):
        H, W = self.hw_dims
        N = x.shape[0]
        if query is not None:
            pos_emb = query.permute(0,2,1).reshape(N, -1, H, W).contiguous()
        elif self.query is not None:
            pos_emb = self.query.weight.view(1,H,W,self.teacher_dims).permute(0,3,1,2).repeat(N,1,1,1)
        else:
            raise NotImplementedError("There is no query!")
       
        preds_S = self.proj_student(x) + self.pos_embed
        pos_emb = self.proj_pos(pos_emb)
        pos_emb = torch.flatten(pos_emb.permute(0, 2, 3, 1), 1, 2)

        fea_S = self.pos_attention(torch.flatten(preds_S.permute(0, 2, 3, 1), 1, 2), pos_emb)
        fea_S = self.ffn(self.norm(fea_S))

        return fea_S
    

class WindowMultiheadPosAttention(nn.Module):
    """Multi-head Attention Module with window partitioning."""
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_shapes=(1,1),
                 input_dims=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 softmax_scale=5.,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 v_shortcut=False,
                 use_layer_scale=False,
                 pos_dims=None
                 ):
        super().__init__()

        self.input_dims = input_dims or embed_dims
        self.pos_dim = pos_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scale = qk_scale or self.head_dims**-0.5
        self.softmax_scale = softmax_scale

        self.q = nn.Linear(self.pos_dim, embed_dims, bias=qkv_bias)
        self.k = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)
        self.v = nn.Linear(self.input_dims, embed_dims, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.out_drop = nn.Dropout(proj_drop)  # Using same dropout rate as proj_drop

        self.window_shapes = window_shapes

        if use_layer_scale:
            self.gamma1 = nn.Parameter(torch.ones(embed_dims) * 1e-2)  # LayerScale implementation
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x, pos_emb):
        # Rest of the forward method remains the same
        B, N, _ = x.shape
        N_out = pos_emb.shape[1]
        N_windows = self.window_shapes[0] * self.window_shapes[1]

        q = self.q(pos_emb).reshape(B, N_out, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, self.head_dims).permute(0, 2, 1, 3)

        if N_windows > 1:
            q = self.separate_tokens(q, self.window_shapes)
            k = self.separate_tokens(k, self.window_shapes)
            v = self.separate_tokens(v, self.window_shapes)

        attn = (q @ k.transpose(-2, -1)) * self.scale * self.softmax_scale
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).view(B, self.num_heads, N_windows, N_out//N_windows, self.head_dims)
        x = x.view(B, self.num_heads, N_out, self.head_dims).transpose(1, 2).reshape(B, N_out, self.embed_dims)

        x = self.proj(x)
        if isinstance(self.gamma1, nn.Parameter):
            x = self.out_drop(self.proj_drop(x) * self.gamma1)
        else:
            x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x
    

    def separate_tokens(self, x, window_shapes=(2,2)):
        BS, num_heads, num_tokens, head_dims = x.shape
        H = W = int(math.sqrt(num_tokens))
        num_win_h, num_win_w = window_shapes

        x = x.view(BS, num_heads, num_win_h, H//num_win_h, num_win_w, W//num_win_w, head_dims).permute(0,1,2,4,3,5,6)
        x = x.contiguous().view(BS, num_heads*num_win_h*num_win_w, -1, head_dims)

        return x
    
class DCT():
    def __init__(self, resolution, device, norm=None, bias=False):
        self.resolution = resolution
        self.norm = norm
        self.device = device

        I = torch.eye(self.resolution, device=self.device)
        self.forward_transform = nn.Linear(resolution, resolution, bias=bias).to(self.device)
        self.forward_transform.weight.data = self._dct(I, norm=self.norm).data.t()
        self.forward_transform.weight.requires_grad = False

        self.inverse_transform = nn.Linear(resolution, resolution, bias=bias).to(self.device)
        self.inverse_transform.weight.data = self._idct(I, norm=self.norm).data.t()
        self.inverse_transform.weight.requires_grad = False

    def _dct(self, x, norm=None):
        """
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        x_shape = x.shape
        N = x_shape[-1]
        x = x.contiguous().view(-1, N)

        v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)

        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

        k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == 'ortho':
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*x_shape)
        return V

    def _idct(self, X, norm=None):
        """
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        x_shape = X.shape
        N = x_shape[-1]

        X_v = X.contiguous().view(-1, x_shape[-1]) / 2

        if norm == 'ortho':
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = torch.arange(x_shape[-1], dtype=X.dtype, device=X.device)[None, :] * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
        x = v.new_zeros(v.shape)
        x[:, ::2] += v[:, :N - (N // 2)]
        x[:, 1::2] += v.flip([1])[:, :N // 2]
        return x.view(*x_shape)

    def forward(self, x):
        X1 = self.forward_transform(x)
        X2 = self.forward_transform(X1.transpose(-1, -2))
        return X2.transpose(-1, -2)

    def inverse(self, x):
        X1 = self.inverse_transform(x)
        X2 = self.inverse_transform(X1.transpose(-1, -2))
        return X2.transpose(-1, -2)
    

class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with residual connection."""
    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.0,
                 add_residual=True):
        super(FFN, self).__init__()
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.dropout = dropout
        self.activate = nn.ReLU(inplace=True) 

        layers = nn.ModuleList()
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, feedforward_channels), 
                    self.activate,
                    nn.Dropout(dropout)))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        self.layers = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + out

