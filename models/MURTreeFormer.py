import torch.nn as nn
import torch
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_
from .SwinBlock import *
from .EncoderModules import *
from .DecoderModules import *
from .function import *

import math
import numpy as np
import torch.utils.checkpoint as checkpoint
from mmcv.cnn import ConvModule

from thop import profile
from thop import clever_format

def expand_state_dict(model_dict, state_dict, num_parallel):
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace("module.", "")
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            ln = ".ln_%d" % i
            replace = True if ln in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(ln, "")
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict


class MURTreeFormer(nn.Module):
    def __init__(
        self,
        num_classes=2,
        pretrain_img_size=512,
        patch_size=4,
        ##can change to 6/8
        in_chans=3,
        in_chans_optical=4,
        in_chans_sar = 1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        num_parallel = 2,
        norm_layer=LayerNormParallel,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
        
    ):
        super().__init__()
        self.drop_path_rate = drop_path_rate
        # self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            in_chans_optical=in_chans_optical,
            in_chans_sar=in_chans_sar,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
        )
        self.img_encoder = ImageEncoder(96)
        self.SURM = SURM_Module(128,128,True)
        self.layers = nn.ModuleList()
        
        

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                num_parallel = 2,
                norm_layer=norm_layer,
                downsample = PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)
        
        self.embedding_dim = 256
        self.feature_strides = [4, 8, 16, 32]
        self.num_parallel = num_parallel

        self.alpha = nn.Parameter(torch.ones(self.num_parallel, requires_grad=True))
        self.num_classes = num_classes
        self.in_channels = [96, 192, 384, 768]

        self.cdm = CDMBlock(192)
        self.decoder = ProgressiveDecoder( num_classes=self.num_classes)


    def forward(self, x):
        opt = x[:,:4]
        sar = x[:,4]
        sar = torch.unsqueeze(sar,1)
        del x
        x = [opt,sar]
        ori_opt = x[0].clone()
        x = self.patch_embed(x)
        opt = x[0]
        sar = x[1]
        Wh, Ww = opt.size(2), opt.size(3)

        opt,sar,reconstruct_loss,kl_loss,uncertainty_map = self.SURM(opt,sar)
        x = [opt,sar]

        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)

                out = [
                    x_out[j]
                    .view(-1, H, W, self.num_features[i])
                    .permute(0, 3, 1, 2)
                    .contiguous()
                    for j in range(len(x_out))
                ]
                outs[i] = out
                if i==1:
                    loss_cdm,diff_map = self.cdm(out[0],out[1])

        new_x0 = []
        new_x1 = []
        for i in range(4):
            new_x0.append(outs[i][0])
            new_x1.append(outs[i][1])
        x = [new_x0, new_x1]
        
        seg_logits, edge_logits = self.decoder([new_x0, new_x1], ori_opt)
        return seg_logits, edge_logits, reconstruct_loss,kl_loss,loss_cdm,diff_map

def analyze_model():
    model = SURM_Module(86,86,True)

    dummy_opt = torch.randn(1, 96, 86, 86)
    dummy_sar = torch.randn(1, 96, 86, 86)
    
    flops, params = profile(model, inputs=(dummy_opt, dummy_sar))
    flops, params = clever_format([flops, params], "%.3f")
    
    print(f"FLOPs: {flops}")
    print(f"param: {params}")
    
    print("\n every layer:")
    flops, params = profile(model, inputs=(dummy_opt, dummy_sar), verbose=True)
    
    

def print_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Info:")
    print(f"All param: {total_params:,}")
    print(f"trained param: {trainable_params:,}")
    print(f"frozen param: {total_params - trainable_params:,}")
    
    # 按层细分
    print("\nevery layer:")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.numel():,}")
        
def run_once():
    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
        return model(inp)
    
def gpu_time_ms(fn, iters=100):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        starter.record()
        _ = fn()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms
    return times

if __name__ == '__main__':
    import os

    model = MURTreeFormer(2).cuda(1)
    optical = torch.randn((1, 5, 512, 512)).cuda(1)
    output1,output2,_,_,_,_ = model(optical)
    
    flops, params = profile(model, inputs=(optical,),verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    
    print(f"FLOPs: {flops}")
    print(f"param: {params}")
    
    
    #calculate SURM
    analyze_model()
    
    from statistics import mean
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MURTreeFormer(2).cuda(1)
    model.eval().to(device)
    
    
    #calculate FPS
    inp = torch.randn((1, 5, 512, 512), device=device)
    
    use_amp = (device == "cuda")
    for _ in range(30):
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            _ = model(inp)
    if device == "cuda":
        torch.cuda.synchronize()

        
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(100):
        starter.record()
        _ = run_once()
        ender.record()
        torch.cuda.synchronize()
        times.append(starter.elapsed_time(ender))  # ms

    times.sort()
    lat_mean = mean(times); lat_p50 = times[len(times)//2]; lat_p95 = times[int(len(times)*0.95)-1]
    print(f"mean {lat_mean:.2f} ms | p50 {lat_p50:.2f} | p95 {lat_p95:.2f}")