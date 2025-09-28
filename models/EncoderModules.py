import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]

class ImageEncoder(nn.Module):
    def __init__(self, input_dim=96, latent_dim=96):
        super(ImageEncoder, self).__init__()
        # self.img_hidden_sz = 2048
        self.dim = 96
        ## 64 = 4*4*4 即 每个patch 4*4 输入为4波段

        self.mu = nn.Linear(input_dim, latent_dim)
        self.logvar = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        # x:5,96,128,128
        x = torch.flatten(x, start_dim=2).transpose(2,1)
        # x:5,16384,96
        mu=self.mu(x)
        logvar=self.logvar(x)
        return mu,logvar,x

class ReconstructDecoder(nn.Module):
    def __init__(self, latent_dim=96, output_dim=96):
        super(ReconstructDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z_sar):
        return self.fc(z_sar)

class PatchEmbed(nn.Module):
    """Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, in_chans_optical=4, in_chans_sar=1, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = ModuleParallel(
            nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        )

        self.proj_optical = nn.Conv2d(in_chans_optical, embed_dim, kernel_size=patch_size, stride=patch_size)
    

        self.proj_sar = nn.Conv2d(in_chans_sar, embed_dim, kernel_size=patch_size, stride=patch_size)
        

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.optical_conv = nn.Conv2d(4,3,1,bias=False)
        self.sar_conv = nn.Conv2d(1,3,1,bias=False)

    def forward(self, x):
        """Forward function."""
        _, _, H, W = x[0].size()
        if W % self.patch_size[1] != 0:
            x[0] = F.pad(x[0], (0, self.patch_size[1] - W % self.patch_size[1]))
            x[1] = F.pad(x[1], (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x[0] = F.pad(x[0], (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
            x[1] = F.pad(x[1], (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x[0] = self.proj_optical(x[0])
        x[1] = self.proj_sar(x[1])

        if self.norm is not None:
            Wh, Ww = x[0].size(2), x[0].size(3)
            for i in range(len(x)):
                x[i] = x[i].flatten(2).transpose(1, 2)
            x = self.norm(x)
            for i in range(len(x)):
                x[i] = x[i].transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x
    
class CrossModalPatchScoringHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, opt_token, sar_token):
        """
        opt_token: [B, HW, C]
        sar_token: [B, HW, C]
        return: [B, HW]  每个patch的不确定性打分
        """
        diff = torch.abs(opt_token - sar_token)  # [B, HW, C]
        score = self.score_mlp(diff).squeeze(-1)  # [B, HW]
        return score

class SURM_Module(nn.Module):
    def __init__(self,opt_token,sar_token,is_training=True):
        super(SURM_Module,self).__init__()
        self.opt = opt_token
        self.sar = sar_token
        self.img_encoder = ImageEncoder(96)
        self.reconstruct_token = ReconstructDecoder(96,96)
        self.gamma = 0.5
        self.k = 500 ##top-K patches
        self.training = is_training
        self.scorer = CrossModalPatchScoringHead(in_dim=96)
        self.beta_kl = 1.0
        self.beta_align = 1.0

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)   # calculate mu
        epsilon = torch.randn_like(std) # random sample
        z = mu + std * epsilon          # reparameterize
        return z
    
    def cog_uncertainty_sample(self, mu, logvar, sample_times=10):
        std = torch.exp(0.5 * logvar)
        mu_expand = mu.unsqueeze(1).repeat(1, sample_times, 1)
        std_expand = std.unsqueeze(1).repeat(1, sample_times, 1)

        epsilon = torch.randn_like(std_expand)
        z_sample = mu_expand + epsilon * std_expand
        return z_sample

    def forward(self,opt_token,sar_token):
        B, C, H, W = sar_token.shape
        patch = H*W
        opt_mu,opt_logvar,opt_token = self.img_encoder(opt_token)
        sar_mu,sar_logvar,sar_token = self.img_encoder(sar_token)

        sar_logvar = torch.clamp(sar_logvar, min=-10.0, max=10.0)
        

        # method1
        # calcalate patch score
        opt_var=torch.exp(opt_logvar)
        sar_var=torch.exp(sar_logvar)
        opt_entropy = torch.prod(opt_var, dim=-1)  # [B, N]
        sar_entropy = torch.prod(sar_var, dim=-1)  # [B, N]
        v = 0.5 * torch.log((sar_entropy + 1e-6) / (opt_entropy + 1e-6))  # [B, N]
        raw_score = self.fc1(v)
        raw_score = self.relu(raw_score)
        raw_score = self.fc2(raw_score)
        patch_scores = torch.softmax(raw_score, dim=-1)
        
        # method2
        # calcalate patch score
        # patch_scores = self.scorer(opt_token, sar_token)  # [B, HW]
        
        
        uncertainty_map = patch_scores.view(B, 1, H, W)
        k_per_img = self.k // B
        topk_idx_all = []
        for i in range(B):
            scores = patch_scores[i]
            _, topk = torch.topk(scores, k=k_per_img)
            topk_idx_all.append((i * patch + topk).to(opt_token.device))
        topk_indices = torch.cat(topk_idx_all)

        bsz_indices = topk_indices // patch
        patch_indices = topk_indices % patch

        prepared_mu = opt_mu[bsz_indices, patch_indices, :]
        prepared_logvar = opt_logvar[bsz_indices, patch_indices, :]
        z = self.reparameterize(prepared_mu, prepared_logvar)
        reconstructed = self.reconstruct_token(z)
        
        sar_flat_updated = sar_token.clone()
        sar_flat_updated[bsz_indices, patch_indices, :] = (
            (1 - self.gamma) * reconstructed + self.gamma * sar_flat_updated[bsz_indices, patch_indices, :]
        )
        
        diff = F.mse_loss(sar_flat_updated, sar_token, reduction='none')  # shape: [B, N, C]
        diff_per_patch = diff.mean(dim=2)  # → [B, N]
        diff_map = diff_per_patch.view(B, 1, H, W)  # reshape


        #sar_token 5,16384,96

        recon_loss = F.mse_loss(
            sar_flat_updated[bsz_indices, patch_indices, :],
            opt_token[bsz_indices, patch_indices, :]
        )
        kl_loss = -0.5 * torch.sum(1 + prepared_logvar - prepared_mu.pow(2) - prepared_logvar.exp())
        kl_loss = kl_loss / B  # normalized
        
        align_kl = F.kl_div(
            input=torch.log_softmax(sar_mu, dim=-1),
            target=torch.softmax(opt_mu.detach(), dim=-1),
            reduction='batchmean'
        )
        
        total_kl_loss = self.beta_kl * kl_loss + self.beta_align * align_kl

        return opt_token, sar_flat_updated, recon_loss, total_kl_loss, uncertainty_map  

class CDMBlock(nn.Module):
    """
    Cross-modal Distillation Module (CDM Block)
    Aligns features from different modalities (e.g., SAR and OPT) to reduce modality gaps.
    """

    def __init__(self, dim, projection=True, metric='cosine'):
        """
        Args:
            dim (int): Dimension of input tokens.
            projection (bool): Whether to project features before alignment.
            metric (str): Distance metric ('cosine', 'mse', 'kl').
        """
        super(CDMBlock, self).__init__()
        self.projection = projection
        self.metric = metric

        if projection:
            self.opt_proj = nn.Linear(dim, dim)
            self.sar_proj = nn.Linear(dim, dim)

    def forward(self, opt_token, sar_token):
        
        B, C, H, W = opt_token.shape
        opt_token = opt_token.flatten(2).transpose(1, 2)  # [B, H*W, C]
        sar_token = sar_token.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        with torch.no_grad():
            raw_diff = (opt_token - sar_token).pow(2).mean(dim=-1)  # [B, N]
            B, N = raw_diff.shape
            H = W = int(N**0.5)  
            raw_diff_map = raw_diff.view(B, H, W)
            raw_diff_map = torch.unsqueeze(raw_diff_map,1)
            # [B, H, W]


        if self.projection:
            opt_feature = self.opt_proj(opt_token)
            sar_feature = self.sar_proj(sar_token)
        else:
            opt_feature = opt_token
            sar_feature = sar_token

        if self.metric == 'cosine':
            cdm_loss = 1 - F.cosine_similarity(opt_feature, sar_feature, dim=-1).mean()
            diff = (opt_feature - sar_feature).pow(2).mean(dim=-1) 
            ##for show the similarity of different modal feature
            diff_map = diff.view(B, H, W)
            diff_map = torch.unsqueeze(diff_map,1)
        elif self.metric == 'mse':
            cdm_loss = F.mse_loss(opt_feature, sar_feature)
        elif self.metric == 'kl':
            opt_logp = F.log_softmax(opt_feature, dim=-1)
            sar_p = F.softmax(sar_feature, dim=-1)
            cdm_loss = F.kl_div(opt_logp, sar_p, reduction='batchmean')
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        return cdm_loss,diff_map