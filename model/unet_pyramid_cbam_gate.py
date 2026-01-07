import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# --- CÁC MODULE PHỤ TRỢ (Giữ nguyên logic của bạn) ---

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=True)

    def forward(self, x):
        avg_out = self.fc2(F.relu(self.fc1(self.avg_pool(x)), inplace=True))
        max_out = self.fc2(F.relu(self.fc1(self.max_pool(x)), inplace=True))
        out = avg_out + max_out
        return x * torch.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attn = torch.sigmoid(self.conv1(concat))
        return x * attn

class CBAM(nn.Module):
    def __init__(self, channels, ratio=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class AttentionGatingBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, inter_channels):
        super(AttentionGatingBlock, self).__init__()
        self.W_g = nn.Conv2d(in_channels_g, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(in_channels_x, inter_channels, kernel_size=2, stride=2)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.final_conv = nn.Conv2d(in_channels_x, in_channels_x, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels_x)

    def forward(self, x, g):
        # Resize g to match x's spatial dimension for calculation if needed (usually handled by stride)
        # But in standard AG, W_x reduces x size to g size, or upsample g to x.
        # Your original code logic:
        theta_x = self.W_x(x) # Downsample x to match g
        
        # Check size mismatch handle
        if theta_x.shape[2:] != g.shape[2:]:
            g = F.interpolate(g, size=theta_x.shape[2:], mode='bilinear', align_corners=True)
            
        phi_g = self.W_g(g)
        
        concat_xg = F.relu(theta_x + phi_g)
        psi = torch.sigmoid(self.psi(concat_xg))

        # Upsample psi back to x size
        upsample_psi = F.interpolate(psi, size=x.shape[2:], mode='bilinear', align_corners=True)
        y = x * upsample_psi
        out = self.final_conv(y)
        return self.bn(out)

# Block Conv đơn giản cho Decoder
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.5):
        super(DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# --- MODEL CHÍNH VỚI RESNET BACKBONE ---

class PyramidCbamGateResNetUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, backbone_name='resnet34', deep_supervision=True, dropout_prob=0.5):
        super(PyramidCbamGateResNetUNet, self).__init__()
        # 2. Pyramid Input Scaling Layers
        self.deep_supervision = deep_supervision
        # Sửa kernel size và stride của AvgPool để downsample mạnh hơn cho khớp
        # Layer 1 của ResNet là H/4 (160x160) -> Cần pool input 640 xuống 160 (chia 4)
        self.pool_scale2 = nn.AvgPool2d(4, 4) 
        # Layer 2 của ResNet là H/8 (80x80) -> Cần pool input 640 xuống 80 (chia 8)
        self.pool_scale3 = nn.AvgPool2d(8, 8)

        # Layer 3 của ResNet là H/16 (40x40) -> Cần pool input 640 xuống 40 (chia 16)
        self.pool_scale4 = nn.AvgPool2d(16, 16)
        # 1. Load Pre-trained ResNet
        # Ta dùng ResNet34 vì nó cân bằng tốt giữa tốc độ và hiệu suất
        # feature channels: [64, 64, 128, 256, 512] (layer0...layer4)
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Sửa lớp đầu vào nếu in_channels != 3
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Trích xuất các tầng của ResNet
        self.encoder0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu) # Stride 2 -> Output: 64ch
        self.encoder1 = self.backbone.maxpool # Stride 2 -> Output: 64ch (nhưng kích thước giảm)
        self.encoder2 = self.backbone.layer1 # 64 ch
        self.encoder3 = self.backbone.layer2 # 128 ch
        self.encoder4 = self.backbone.layer3 # 256 ch
        self.center   = self.backbone.layer4 # 512 ch

        # 2. Pyramid Input Scaling Layers
        # Để trộn ảnh thu nhỏ vào Feature Map của ResNet, ta cần Conv 1x1 để khớp số kênh
        # ResNet channels tại các điểm nối: 
        # Layer1 (64), Layer2 (128), Layer3 (256)
        self.avgpool = nn.AvgPool2d(2, 2)
        
        # Conv để chiếu ảnh input (3 ch) vào feature map của ResNet
        self.scale2_conv = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)  # Khớp Layer 1
        self.scale3_conv = nn.Conv2d(in_channels, 128, kernel_size=3, padding=1) # Khớp Layer 2
        self.scale4_conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1) # Khớp Layer 3

        # 3. CBAM Modules (Áp dụng sau mỗi Feature Map của ResNet)
        self.cbam1 = CBAM(64)  # Sau layer 1
        self.cbam2 = CBAM(128) # Sau layer 2
        self.cbam3 = CBAM(256) # Sau layer 3
        self.cbam4 = CBAM(512) # Sau Center (Layer 4)

        # 4. Attention Gating Blocks
        # Gating Signal đi từ dưới lên (Coarse), Skip Connection đi ngang (Fine)
        # AG1: Gate=Center(512), X=Encoder4(256)
        self.attgating1 = AttentionGatingBlock(in_channels_x=256, in_channels_g=512, inter_channels=128)
        
        # AG2: Gate=Up1(256), X=Encoder3(128)
        self.attgating2 = AttentionGatingBlock(in_channels_x=128, in_channels_g=256, inter_channels=64)
        
        # AG3: Gate=Up2(128), X=Encoder2(64)
        self.attgating3 = AttentionGatingBlock(in_channels_x=64, in_channels_g=128, inter_channels=32)

        # 5. Decoder Path
        # Up1: Center(512) -> Up(256) + Attn1(256) -> Conv(256)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = DecoderBlock(256 + 256, 256, dropout_prob=dropout_prob)
        
        # Up2: Dec1(256) -> Up(128) + Attn2(128) -> Conv(128)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DecoderBlock(128 + 128, 128, dropout_prob=dropout_prob) 
        
        # Up3: Dec2(128) -> Up(64) + Attn3(64) -> Conv(64)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = DecoderBlock(64 + 64, 64, dropout_prob=dropout_prob)   
        # Up4: Dec3(64) -> Up(64) + Encoder0(64) -> Conv(32)
        # Encoder0 là lớp stem đầu tiên, kích thước lớn nhất
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec4 = DecoderBlock(64 + 64, 32, dropout_prob=dropout_prob)  

        # Final Conv
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        # --- [THÊM MỚI] DEEP SUPERVISION HEADS ---
        if self.deep_supervision:
            # Các đầu ra phụ từ d1, d2, d3
            # d1 (256ch), d2 (128ch), d3 (64ch)
            self.seg_head_d1 = nn.Conv2d(256, out_channels, kernel_size=1)
            self.seg_head_d2 = nn.Conv2d(128, out_channels, kernel_size=1)
            self.seg_head_d3 = nn.Conv2d(64, out_channels, kernel_size=1)
        # ----------------------------------------
    def forward(self, x):
        # --- PYRAMID INPUTS ---
        # Tạo các phiên bản thu nhỏ của ảnh đầu vào
        # x_scale2 = self.avgpool(x)        # /2
        # x_scale3 = self.avgpool(x_scale2) # /4
        # x_scale4 = self.avgpool(x_scale3) # /8
        x_scale2 = self.pool_scale2(x) # 640 -> 160 (Khớp e1)
        x_scale3 = self.pool_scale3(x) # 640 -> 80  (Khớp e2)
        x_scale4 = self.pool_scale4(x) # 640 -> 40  (Khớp e3)
        # --- ENCODER (RESNET) + PYRAMID FUSION ---
        # Stem: Conv1 -> BN -> ReLU
        e0 = self.encoder0(x) # (B, 64, H/2, W/2)
        e0_pool = self.encoder1(e0) # MaxPool -> (B, 64, H/4, W/4)
        
        # Layer 1 (64ch)
        # Trộn ảnh input scale2 vào đây (Fusion)
        feat_scale2 = self.scale2_conv(x_scale2)
        # ResNet layer1 giữ nguyên kích thước (H/4)
        # Cộng feature map của ResNet với feature map của ảnh input (Residual connection style)
        e1 = self.encoder2(e0_pool) 
        # e1 = e1 + F.interpolate(feat_scale2, size=e1.shape[2:]) # Add fusion
        e1 = e1 + feat_scale2
        e1 = self.cbam1(e1) # Apply CBAM

        # Layer 2 (128ch)
        feat_scale3 = self.scale3_conv(x_scale3)
        e2 = self.encoder3(e1) # Stride 2 -> H/8
        # e2 = e2 + F.interpolate(feat_scale3, size=e2.shape[2:]) # Add fusion
        e2 = e2 + feat_scale3
        e2 = self.cbam2(e2)

        # Layer 3 (256ch)
        feat_scale4 = self.scale4_conv(x_scale4)
        e3 = self.encoder4(e2) # Stride 2 -> H/16
        # e3 = e3 + F.interpolate(feat_scale4, size=e3.shape[2:]) # Add fusion
        e3 = e3 + feat_scale4 # <--- Cộng trực tiếp
        e3 = self.cbam3(e3)

        # Center (Layer 4 - 512ch)
        center = self.center(e3) # Stride 2 -> H/32
        center = self.cbam4(center)

        # --- DECODER + ATTENTION GATING ---
        
        # Up 1
        d1 = self.up1(center) # 512 -> 256
        # Gate: Center, Skip: e3
        attn1 = self.attgating1(x=e3, g=center)
        d1 = torch.cat([d1, attn1], dim=1) # 256 + 256
        d1 = self.dec1(d1)

        # Up 2
        d2 = self.up2(d1) # 256 -> 128
        # Gate: d1, Skip: e2
        attn2 = self.attgating2(x=e2, g=d1)
        d2 = torch.cat([d2, attn2], dim=1) # 128 + 128
        d2 = self.dec2(d2)

        # Up 3
        d3 = self.up3(d2) # 128 -> 64
        # Gate: d2, Skip: e1
        attn3 = self.attgating3(x=e1, g=d2)
        d3 = torch.cat([d3, attn3], dim=1) # 64 + 64
        d3 = self.dec3(d3)

        # Up 4 (Về kích thước H/2)
        d4 = self.up4(d3) # 64 -> 64
        # Skip connection cuối cùng từ e0 (Stem layer)
        # e0 kích thước H/2. 
        if d4.shape[2:] != e0.shape[2:]:
             d4 = F.interpolate(d4, size=e0.shape[2:])
        d4 = torch.cat([d4, e0], dim=1) # 64 + 64
        d4 = self.dec4(d4) # -> 32

        # Final Upsample (Về kích thước H gốc) & Conv
        out = self.final(d4)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        # --- [THÊM MỚI] RETURN LIST NẾU DEEP SUPERVISION ---
        if self.deep_supervision and self.training:
            # Lưu ý: Hàm Loss của bạn tự động resize target xuống, 
            # nên ta KHÔNG cần upsample các output phụ lên kích thước gốc.
            # Để nguyên kích thước nhỏ giúp tiết kiệm bộ nhớ và tính toán nhanh hơn.
            
            deep1 = self.seg_head_d1(d1) # Output từ tầng sâu nhất (nhỏ nhất)
            deep2 = self.seg_head_d2(d2)
            deep3 = self.seg_head_d3(d3)
            
            # Thứ tự return quan trọng: [Final, Phụ_to, Phụ_vừa, Phụ_nhỏ]
            # Tương ứng với weights = [1.0, 0.5, 0.25, 0.1]
            return [out, deep3, deep2, deep1] 
        # ---------------------------------------------------        
        return out
class PyramidCbamGateResNet50UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, backbone_name='resnet50', deep_supervision=True, dropout_prob=0.5):
        super(PyramidCbamGateResNet50UNet, self).__init__()
        self.deep_supervision = deep_supervision
        
        # --- PYRAMID POOLING (Giữ nguyên) ---
        self.pool_scale2 = nn.AvgPool2d(4, 4) 
        self.pool_scale3 = nn.AvgPool2d(8, 8)
        self.pool_scale4 = nn.AvgPool2d(16, 16)

        # 1. LOAD BACKBONE RESNET50
        # Weights mặc định
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Encoder Layers
        self.encoder0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu)
        self.encoder1 = self.backbone.maxpool
        self.encoder2 = self.backbone.layer1 # Output: 256 channels (ResNet34 là 64)
        self.encoder3 = self.backbone.layer2 # Output: 512 channels (ResNet34 là 128)
        self.encoder4 = self.backbone.layer3 # Output: 1024 channels (ResNet34 là 256)
        self.center   = self.backbone.layer4 # Output: 2048 channels (ResNet34 là 512)

        # 2. PYRAMID FUSION (Sửa số kênh đầu ra để khớp Backbone mới)
        self.scale2_conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)  # Khớp Layer 1 (256)
        self.scale3_conv = nn.Conv2d(in_channels, 512, kernel_size=3, padding=1)  # Khớp Layer 2 (512)
        self.scale4_conv = nn.Conv2d(in_channels, 1024, kernel_size=3, padding=1) # Khớp Layer 3 (1024)

        # 3. CBAM (Sửa kênh)
        self.cbam1 = CBAM(256) 
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

        # 4. ATTENTION GATING (Sửa in_channels)
        # AG1: Gate=Center(2048), X=Encoder4(1024) -> Inter=512
        self.attgating1 = AttentionGatingBlock(in_channels_x=1024, in_channels_g=2048, inter_channels=512)
        
        # AG2: Gate=Up1(512), X=Encoder3(512) -> Inter=256
        # Lưu ý: Up1 giảm kênh từ 2048 -> 512
        self.attgating2 = AttentionGatingBlock(in_channels_x=512, in_channels_g=512, inter_channels=256)
        
        # AG3: Gate=Up2(256), X=Encoder2(256) -> Inter=128
        self.attgating3 = AttentionGatingBlock(in_channels_x=256, in_channels_g=256, inter_channels=128)

        # 5. DECODER PATH (Sửa ConvTranspose và DecoderBlock)
        
        # Up 1: Center(2048) -> 512
        self.up1 = nn.ConvTranspose2d(2048, 512, kernel_size=2, stride=2)
        # Cat: Up1(512) + Attn1(1024) = 1536 -> Out 512
        self.dec1 = DecoderBlock(512 + 1024, 512, dropout_prob=dropout_prob)
        
        # Up 2: Dec1(512) -> 256
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Cat: Up2(256) + Attn2(512) = 768 -> Out 256
        self.dec2 = DecoderBlock(256 + 512, 256, dropout_prob=dropout_prob)
        
        # Up 3: Dec2(256) -> 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Cat: Up3(128) + Attn3(256) = 384 -> Out 128
        self.dec3 = DecoderBlock(128 + 256, 128, dropout_prob=dropout_prob)
        
        # Up 4: Dec3(128) -> 64
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Cat: Up4(64) + Encoder0(64) = 128 -> Out 64
        self.dec4 = DecoderBlock(64 + 64, 64, dropout_prob=dropout_prob)

        # Final
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # Deep Supervision Heads (Sửa kênh đầu vào)
        if self.deep_supervision:
            # d1 (512), d2 (256), d3 (128)
            self.seg_head_d1 = nn.Conv2d(512, out_channels, kernel_size=1)
            self.seg_head_d2 = nn.Conv2d(256, out_channels, kernel_size=1)
            self.seg_head_d3 = nn.Conv2d(128, out_channels, kernel_size=1)
            
    # --- PHẦN FORWARD GIỮ NGUYÊN (KHÔNG CẦN SỬA) ---
    def forward(self, x):
        # ... (Copy nguyên hàm forward cũ vào đây) ...
        # Logic forward không thay đổi vì ta chỉ thay đổi kích thước kênh bên trong __init__
        # Chỉ cần đảm bảo copy đủ code forward từ bài cũ
        return super().forward(x) if hasattr(super(), 'forward') else self._forward_impl(x)

    # Copy hàm forward từ code cũ của bạn vào đây và đổi tên thành _forward_impl hoặc forward
    def forward(self, x):
        # --- PYRAMID INPUTS ---
        x_scale2 = self.pool_scale2(x)
        x_scale3 = self.pool_scale3(x)
        x_scale4 = self.pool_scale4(x)

        # --- ENCODER ---
        e0 = self.encoder0(x)
        e0_pool = self.encoder1(e0)
        
        feat_scale2 = self.scale2_conv(x_scale2)
        e1 = self.encoder2(e0_pool) 
        e1 = e1 + feat_scale2
        e1 = self.cbam1(e1)

        feat_scale3 = self.scale3_conv(x_scale3)
        e2 = self.encoder3(e1) 
        e2 = e2 + feat_scale3
        e2 = self.cbam2(e2)

        feat_scale4 = self.scale4_conv(x_scale4)
        e3 = self.encoder4(e2) 
        e3 = e3 + feat_scale4
        e3 = self.cbam3(e3)

        center = self.center(e3)
        center = self.cbam4(center)

        # --- DECODER ---
        d1 = self.up1(center)
        attn1 = self.attgating1(x=e3, g=center)
        d1 = torch.cat([d1, attn1], dim=1)
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        attn2 = self.attgating2(x=e2, g=d1)
        d2 = torch.cat([d2, attn2], dim=1)
        d2 = self.dec2(d2)

        d3 = self.up3(d2)
        attn3 = self.attgating3(x=e1, g=d2)
        d3 = torch.cat([d3, attn3], dim=1)
        d3 = self.dec3(d3)

        d4 = self.up4(d3)
        if d4.shape[2:] != e0.shape[2:]:
             d4 = F.interpolate(d4, size=e0.shape[2:])
        d4 = torch.cat([d4, e0], dim=1)
        d4 = self.dec4(d4)

        out = self.final(d4)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)

        if self.deep_supervision and self.training:
            deep1 = self.seg_head_d1(d1)
            deep2 = self.seg_head_d2(d2)
            deep3 = self.seg_head_d3(d3)
            return [out, deep3, deep2, deep1]
            
        return out
