import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from thop import clever_format


class GradientMagnitudeAttention(nn.Module):
    def __init__(self, lambda_att=2.0):
        super().__init__()
        self.lambda_att = lambda_att
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
    def forward(self, ori_input, feat):
        with torch.no_grad():
            opt = ori_input[:,0,:]/3+ori_input[:,1,:]/3+ori_input[:,2,:]/3
            opt = torch.unsqueeze(opt,dim=1)
            
            opt = 1.0-opt
            lum = opt**5

            if lum.shape[-2:] != feat.shape[-2:]:
                lum = F.interpolate(lum, size=feat.shape[-2:], mode='bilinear', align_corners=False)
            grad_x = F.conv2d(lum, self.sobel_x, padding=1)
            grad_y = F.conv2d(lum, self.sobel_y, padding=1)
            grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
            att = torch.exp(-self.lambda_att * grad_mag).clamp(0, 1)
            att = 1- att

        return feat * att + feat.detach() * (1 - att)



class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ChannelAlign(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ChannelAlign, self).__init__()
        self.use_proj = in_channels != out_channels
        self.se = SEBlock(in_channels)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.proj = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.proj(x)
        x = self.se(x)
        x = self.conv_layers(x)
        return self.relu(x + identity)


class ConvSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSEBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1)
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        x = self.conv(x)
        return self.se(x)

class DecoderRefine(nn.Module):
    def __init__(self, embed_dim=256, num_classes=2):
        super(DecoderRefine, self).__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )
        self.final = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.edge = nn.Conv2d(embed_dim, 1, kernel_size=1)

    def forward(self, x, out_size):
        x = self.up1(x)  # 128→256
        x = self.up2(x)  # 256→512
        seg = F.interpolate(self.final(x), size=out_size, mode='bilinear', align_corners=False)
        edge = F.interpolate(self.edge(x), size=out_size, mode='bilinear', align_corners=False)
        return seg, edge

class ProgressiveDecoder(nn.Module):
    def __init__(self, embed_dim=256,num_classes=2):
        super(ProgressiveDecoder, self).__init__()
        self.embed_dim = 256
        
        self.gam = GradientMagnitudeAttention(lambda_att=2.0)

        self.align_c1 = ChannelAlign(in_channels=192, mid_channels=128, out_channels=128)
        self.align_c2 = ChannelAlign(in_channels=384, mid_channels=256, out_channels=256)
        self.align_c3 = ChannelAlign(in_channels=768, mid_channels=384, out_channels=256)
        self.align_c4 = ChannelAlign(in_channels=1536, mid_channels=768, out_channels=256)

        self.conv3 = ConvSEBlock(in_channels=256 + 256, out_channels=256)
        self.conv2 = ConvSEBlock(in_channels=256 + 256, out_channels=256)
        self.conv1 = ConvSEBlock(in_channels=128 + 256, out_channels=256)
        
        self.decoder_refine = DecoderRefine(embed_dim,num_classes)

        self.final = nn.Conv2d(256, num_classes, kernel_size=1)
        self.edge_head = nn.Conv2d(256, 1, kernel_size=1)
    
    def analyze_model(self,x1, x2):
        model = DecoderRefine(256,2).cuda(1)
        flops, params = profile(model, inputs=(x1, 512,))
        flops, params = clever_format([flops, params], "%.3f")
        
        print(f"FLOPs: {flops}")
        print(f"param: {params}")
        

    def forward(self, features, opt_ori):
        opt_stage1, opt_stage2, opt_stage3, opt_stage4 = features[0]
        sar_stage1, sar_stage2, sar_stage3, sar_stage4 = features[1]

        c1 = self.align_c1(torch.cat([opt_stage1, sar_stage1], dim=1))  # [B, 128, H, W]
        c2 = self.align_c2(torch.cat([opt_stage2, sar_stage2], dim=1))  # [B, 256, H/2, W/2]
        c3 = self.align_c3(torch.cat([opt_stage3, sar_stage3], dim=1))  # [B, 256, H/4, W/4]
        c4 = self.align_c4(torch.cat([opt_stage4, sar_stage4], dim=1))  # [B, 256, H/8, W/8]

        # Decoder blocks
        x = F.interpolate(c4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, c3], dim=1)
        x = self.gam(opt_ori, x)  # <-- apply GAM here
        x = self.conv3(x)
        # x = self.conv3(torch.cat([x, c3], dim=1))

        x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv2(torch.cat([x, c2], dim=1))

        x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv1(torch.cat([x, c1], dim=1))


        seg, edge = self.decoder_refine(x, opt_ori.shape[-2:])
        return seg, edge
