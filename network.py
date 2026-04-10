import torch
from torch import nn
import torch.nn.functional as F
from backbone.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

from torch.profiler import profile, record_function, ProfilerActivity

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)
    
class RGBTfusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_gate = nn.Linear(256, 256, bias=False)
        self.thermal_gate = nn.Linear(256, 256, bias=False)
        
    def forward(self, rgb_cls, rgb_patch, thermal_cls, thermal_patch):
        rgb_cls = rgb_cls.unsqueeze(1)
        thermal_cls = thermal_cls.unsqueeze(1)
        attn_score_rgb = torch.bmm(rgb_cls, rgb_patch.permute(0, 2, 1)).squeeze(1) # [B, 256]
        attn_score_thermal = torch.bmm(thermal_cls, thermal_patch.permute(0, 2, 1)).squeeze(1)

        w_rgb = self.rgb_gate(attn_score_rgb) # w_rgb: [B, 256]
        w_thermal = self.thermal_gate(attn_score_thermal)

        # L2 normalization
        w = torch.cat((w_rgb, w_thermal), dim=1)
        w = F.normalize(w, p=2, dim=1)
        w_rgb, w_thermal = w[:, :256], w[:, 256:]

        if self.eval():
            self.w_rgb_raw = w_rgb
            self.w_thermal_raw = w_thermal
            self.f_rgb_raw = rgb_patch
            self.f_thermal_raw = thermal_patch
            self.attn_score_rgb = attn_score_rgb
            self.attn_score_thermal = attn_score_thermal
               
        w = torch.cat((w_rgb, w_thermal), dim=1) # [B, 512]
        
        # filter
        threshold = w.mean(dim=1, keepdim=True) # [B, 1]
        w = torch.where(w < threshold, torch.zeros_like(w), w).unsqueeze(-1) # [B, 512, 1]

        w_rgb = w[:, :256, :] # [B, 256, 1]
        w_thermal = w[:, 256:, :] # [B, 256, 1]

        x_fused = w_rgb * rgb_patch + w_thermal * thermal_patch # [B, 256, 768]
        
        if self.eval():
            self.w_rgb = w_rgb
            self.w_thermal = w_thermal
            self.f_fused = x_fused

        return x_fused

class RGBTVPR_Net(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer."""
    def __init__(self, pretrained_foundation = False, foundation_model_path = None):
        super().__init__()

        self.rgb_backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.thermal_backbone = get_backbone(pretrained_foundation, foundation_model_path)

        self.fusion = RGBTfusion()
        self.aggregation = nn.Sequential(L2Norm(), GeM(work_with_tokens=None), Flatten())

       
    def forward(self, x, query_flags: torch.Tensor):
        # x: [B, 6, W, H]
        thermal_out = self.thermal_backbone(x[:, 3:, :, :])
        thermal_cls = thermal_out["x_norm_clstoken"]
        thermal_patch = thermal_out["x_norm_patchtokens"]

        _, P, D = thermal_patch.shape  # P=256, D=768

        # Database 항목: thermal_patch만 사용
        database_x = thermal_patch[~query_flags]  # [N_db, 256, 768]

        # rgb backbone은 query 항목에만 실행 (database는 RGB 미사용)
        rgb_out = self.rgb_backbone(x[query_flags, :3, :, :])
        queries_rgb_cls = rgb_out["x_norm_clstoken"]
        queries_rgb_patch = rgb_out["x_norm_patchtokens"]
        queries_thermal_cls = thermal_cls[query_flags]
        queries_thermal_patch = thermal_patch[query_flags]

        # fusion: query 항목에만 적용
        queries_x = self.fusion(
            queries_rgb_cls, queries_rgb_patch,
            queries_thermal_cls, queries_thermal_patch,
        )

        full_x = torch.cat([queries_x, database_x], dim=0)
        full_x = full_x.permute(0, 2, 1)

        full_x = full_x.view(-1, D, 16, 16)
        full_x = self.aggregation(full_x)  # [B, 768]

        return full_x
    
def get_backbone(pretrained_foundation, foundation_model_path):
    backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0)
    if pretrained_foundation:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone
