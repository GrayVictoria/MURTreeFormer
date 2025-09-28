import torch



def reparameterise(mu, std):
    """
    mu : [batch_size,z_dim]
    std : [batch_size,z_dim]        
    """        
    # get epsilon from standard normal
    eps = torch.randn_like(std)
    return mu + std*eps

def cog_uncertainty_sample1(opt_mu, opt_var, sar_mu, sar_var, sample_times=10):
        #生成采样
        l_list = []
        for _ in range(sample_times):
            l_list.append(reparameterise(opt_mu, opt_var))
        l_sample = torch.stack(l_list, dim=2)
        # bsz*samples*dim 819200

        v_list = []
        for _ in range(sample_times):
            v_list.append(reparameterise(sar_mu, sar_var))
        v_sample = torch.stack(v_list, dim=2)
        
        return l_sample, v_sample

def cog_uncertainty_normal(unc_dict, normal_type="None"):

    key_list = [k for k, _ in unc_dict.items()]
    comb_list = [t for _, t in unc_dict.items()]
    comb_t = torch.stack(comb_list, dim=1)
    mat = torch.exp(torch.reciprocal(comb_t))
    mat_sum = mat.sum(dim=-1, keepdim=True)
    weight = mat / mat_sum

    if normal_type == "minmax":
        weight = weight / torch.max(weight, dim=1)[0].unsqueeze(-1)  # [bsz, mod_num]
        for i, key in enumerate(key_list):
            unc_dict[key] = weight[:, i]
    else:
        pass
        # raise TypeError("Unsupported Operations at cog_uncertainty_normal!")

    return unc_dict

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x