import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import segmentation_models_pytorch as smp

# ==========================================
# 1. CÁC HÀM TÍNH TOÁN CỐT LÕI (CORE FUNCTIONS)
# ==========================================
def to_numpy(tensor):
    return tensor.cpu().detach().item()

def dice_coeff(logits, target, epsilon=1e-6):
    probs = torch.sigmoid(logits)
    # Flatten: [B, C, H, W] -> [B, -1]
    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)
    
    numerator = 2 * (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)
    dice = (numerator + epsilon) / (denominator + epsilon)
    return dice # [Batch_size]

def dice_coef_loss_per_image(logits, targets):
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    dice = dice_coeff(logits, targets)
    loss = 1.0 - dice
    return loss # [Batch_size]

def binary_focal_loss_with_logits(logits, targets, alpha=None, gamma=2.0, reduction="mean", eps=1e-7):
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)
    pt = pt.clamp(min=eps, max=1.0 - eps)
    focal_factor = (1.0 - pt) ** gamma

    if alpha is not None:
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    else:
        alpha_t = 1.0

    loss = alpha_t * focal_factor * bce

    if reduction == "mean": return loss.mean()
    elif reduction == "sum": return loss.sum()
    else: return loss

# ==========================================
# 2. CÁC CLASS LOSS (WRAPPER ĐỂ DÙNG TRONG TRAINER)
# ==========================================

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, logits, targets):
        if isinstance(logits, (list, tuple)): logits = logits[0]
        # Trả về scalar mean loss
        return dice_coef_loss_per_image(logits, targets).mean()

class HybricLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
    def forward(self, logits, targets):
        if isinstance(logits, (list, tuple)): logits = logits[0]
        # Hybric = Dice + 0.5 * Focal (alpha=None)
        focal = binary_focal_loss_with_logits(logits, targets, alpha=None, gamma=2.0, reduction="mean")
        return self.dice_loss(logits, targets) + 0.5 * focal

# class BCEDiceLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss()
#     def forward(self, logits, targets):
#         if targets.ndim == 3: targets = targets.unsqueeze(1)
#         bce = self.bce(logits, targets)
#         dice = dice_coef_loss_per_image(logits, targets).mean()
#         return bce + dice
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        # Dùng BCEWithLogitsLoss là chuẩn xác nhất
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if isinstance(logits, (list, tuple)):
            logits = logits[0] # Chỉ tính loss cho output chính
        # 1. Đảm bảo targets đúng chiều [Batch, 1, H, W]
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
            
        # 2. Tính BCE Loss
        # logits: [B, 1, H, W], targets: [B, 1, H, W]
        bce_loss = self.bce(logits, targets)
        
        # 3. Tính Dice Loss
        # Lưu ý: Hàm dice_coef_loss_per_image của bạn đã tự sigmoid rồi
        dice_loss = dice_coef_loss_per_image(logits, targets).mean()
        
        # 4. Cộng gộp (Weighted Sum)
        # Công thức chuẩn: 0.5 * BCE + 0.5 * Dice
        loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss
        
        return loss
class BCEWeightLoss(nn.Module):
    def __init__(self, pos_weight=100.0):
        super().__init__()
        self.pos_weight = torch.tensor(pos_weight)
    def forward(self, logits, targets):
        if isinstance(logits, (list, tuple)):
            logits = logits[0] # Chỉ tính loss cho output chính
        if targets.ndim == 3: targets = targets.unsqueeze(1)
        # Đảm bảo pos_weight nằm cùng device với logits
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        if isinstance(logits, (list, tuple)):
            logits = logits[0] # Chỉ tính loss cho output chính
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        TP = (probs * targets).sum()
        FP = ((1 - targets) * probs).sum()
        FN = (targets * (1 - probs)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1.0 - tversky

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, ce_ratio=0.5, focal_gamma=2.0, deep_supervision=True):
        super().__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.focal_gamma = focal_gamma
        self.deep_supervision = deep_supervision
        # Định nghĩa trọng số cho các output (Giống FocalTversky)
        # Output chính (index 0) quan trọng nhất
        self.ds_weights = [1.0, 0.5, 0.25, 0.1] 

    def forward(self, logits, targets):
        """
        Combo Loss = Ratio * Focal + (1 - Ratio) * Dice
        Hỗ trợ Deep Supervision (List input)
        """
        # --- TRƯỜNG HỢP 1: DEEP SUPERVISION (Logits là List) ---
        if isinstance(logits, (list, tuple)):
            total_loss = 0
            # Tính tổng trọng số để normalize (như bạn đã làm bên FocalTversky)
            # Logic: Lấy danh sách weight tương ứng với số lượng output thực tế
            current_weights_len = min(len(logits), len(self.ds_weights))
            current_weights = self.ds_weights[:current_weights_len]
            
            # Nếu output nhiều hơn weight định sẵn, cộng thêm phần dư 0.05 vào tổng
            if len(logits) > len(self.ds_weights):
                sum_w = sum(current_weights) + (len(logits) - len(self.ds_weights)) * 0.05
            else:
                sum_w = sum(current_weights)

            # Duyệt qua từng output
            for i, logit in enumerate(logits):
                # Lấy trọng số w
                w = self.ds_weights[i] if i < len(self.ds_weights) else 0.05
                # Normalize
                w = w / sum_w
                
                # Resize targets nếu cần (cho các tầng sâu kích thước nhỏ)
                if logit.shape[-2:] != targets.shape[-2:]:
                     target_resized = F.interpolate(targets, size=logit.shape[-2:], mode='nearest')
                else:
                     target_resized = targets
                
                # --- TÍNH LOSS THÀNH PHẦN ---
                # 1. Dice
                dice_loss = dice_coef_loss_per_image(logit, target_resized).mean()
                # 2. Focal
                focal_loss = binary_focal_loss_with_logits(
                    logit, target_resized, 
                    alpha=self.alpha, gamma=self.focal_gamma, reduction="mean"
                )
                
                # Tổng hợp Combo cho tầng này
                layer_loss = self.ce_ratio * focal_loss + (1 - self.ce_ratio) * dice_loss
                
                # Cộng dồn vào tổng loss với trọng số w
                total_loss += w * layer_loss
            
            return total_loss

        # --- TRƯỜNG HỢP 2: BÌNH THƯỜNG (Logits là Tensor đơn) ---
        else:
            dice_loss = dice_coef_loss_per_image(logits, targets).mean()
            focal_loss = binary_focal_loss_with_logits(
                logits, targets, alpha=self.alpha, gamma=self.focal_gamma, reduction="mean"
            )
            return self.ce_ratio * focal_loss + (1 - self.ce_ratio) * dice_loss
# class ComboLoss(nn.Module):
#     def __init__(self, alpha=0.5, ce_ratio=0.5, focal_gamma=2.0):
#         super().__init__()
#         self.alpha = alpha
#         self.ce_ratio = ce_ratio
#         self.focal_gamma = focal_gamma

#     def forward(self, logits, targets):
#         if isinstance(logits, (list, tuple)):
#             logits = logits[0] # Chỉ tính loss cho output chính
#         dice_loss = dice_coef_loss_per_image(logits, targets).mean()
#         focal_loss = binary_focal_loss_with_logits(
#             logits, targets, alpha=self.alpha, gamma=self.focal_gamma, reduction="mean"
#         )
#         return self.ce_ratio * focal_loss + (1 - self.ce_ratio) * dice_loss

# ==========================================
# 3. FOCAL TVERSKY LOSS & GLOBAL INSTANCE
# ==========================================

# class FocalTverskyLoss(nn.Module):
#     def __init__(self, alpha=0.4, beta=0.6, gamma=1.33, smooth=1e-6):
#         super().__init__()
#         self.tversky = TverskyLoss(alpha, beta, smooth)
#         self.gamma = gamma

#     def forward(self, logits, targets):
#         tversky_loss = self.tversky(logits, targets)
#         return torch.pow(tversky_loss, self.gamma)

#     def update_params(self, alpha=None, beta=None, gamma=None):
#         """Cập nhật nóng tham số"""
#         if alpha is not None:
#             self.tversky.alpha = alpha
#         if beta is not None:
#             self.tversky.beta = beta
#         if gamma is not None:
#             self.gamma = gamma
#         print(f"[LOSS UPDATE] Changed params to: alpha={self.tversky.alpha}, beta={self.tversky.beta}, gamma={self.gamma}")

# --- GLOBAL INSTANCE (Để dùng cho chiến lược 3 giai đoạn) ---
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.4, beta=0.6, gamma=1.33, smooth=1e-6, deep_supervision=True):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma
        self.deep_supervision = deep_supervision
        # Trọng số cho Deep Supervision (giảm dần từ output chính đến output phụ)
        # Output 0 (Final): 1.0
        # Output 1: 0.5
        # Output 2: 0.1
        # Output 3: 0.05
        # self.ds_weights = [1.0, 0.5, 0.1, 0.05]
        self.ds_weights = [1.0, 0.5, 0.25, 0.1] # Tinh chỉnh nhẹ cho EfficientNet
    def forward(self, logits, targets):
        """
        Hàm forward thông minh: Tự động phát hiện logits là Single Tensor hay List
        """
        # --- TRƯỜNG HỢP 1: DEEP SUPERVISION (Logits là List) ---
        if isinstance(logits, (list, tuple)):
            total_loss = 0
            # Tính tổng trọng số để normalize (nếu muốn loss ổn định)
            current_weights_len = min(len(logits), len(self.ds_weights))
            current_weights = self.ds_weights[:current_weights_len]
            if len(logits) > len(self.ds_weights):
                sum_w = sum(current_weights) + (len(logits) - len(self.ds_weights)) * 0.05
            else:
                sum_w = sum(current_weights)
            # Duyệt qua từng output trong list
            for i, logit in enumerate(logits):
                # Lấy trọng số tương ứng (nếu vượt quá list weights thì lấy cái cuối cùng)
                w = self.ds_weights[i] if i < len(self.ds_weights) else 0.05
                # Normalize (Optional): w = w / sum_w
                w = w / sum_w
                # Resize targets nếu kích thước không khớp (chỉ xảy ra ở tầng rất sâu)
                if logit.shape[-2:] != targets.shape[-2:]:
                     target_resized = F.interpolate(targets, size=logit.shape[-2:], mode='nearest')
                else:
                     target_resized = targets
                # Tính Loss thành phần
                tversky_loss = self.tversky(logit, target_resized)
                focal_loss = torch.pow(tversky_loss, self.gamma)
                
                # Cộng dồn vào tổng (có nhân trọng số)
                total_loss += w * focal_loss
            
            return total_loss

        # --- TRƯỜNG HỢP 2: BÌNH THƯỜNG (Logits là Tensor đơn) ---
        else:
            tversky_loss = self.tversky(logits, targets)
            return torch.pow(tversky_loss, self.gamma)

    def update_params(self, alpha=None, beta=None, gamma=None):
        """Cập nhật nóng tham số"""
        if alpha is not None:
            self.tversky.alpha = alpha
        if beta is not None:
            self.tversky.beta = beta
        if gamma is not None:
            self.gamma = gamma
        print(f"[LOSS UPDATE] Changed params to: alpha={self.tversky.alpha}, beta={self.tversky.beta}, gamma={self.gamma}")
_focal_tversky_global = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)

# ==========================================
# 4. FACTORY FUNCTION (HÀM LẤY LOSS) - ĐÃ SỬA BUG
# ==========================================

def get_loss_instance(loss_name):
    """
    Router trả về OBJECT hàm loss (đã khởi tạo).
    KHÔNG truyền logits/targets vào đây.
    """
    # 1. FocalTversky: Trả về biến toàn cục để có thể update params
    if loss_name == "FocalTversky_loss":
        return _focal_tversky_global
    # 2. Tversky thường: Tạo mới
    elif loss_name == "Tversky_loss":
        return TverskyLoss(alpha=0.3, beta=0.7)
    # 3. Combo Loss
    elif loss_name == "Combo_loss": 
        return ComboLoss(alpha=0.8, ce_ratio=0.5, focal_gamma=2.0)
    # 4. Các loss khác (Đã wrap thành Class)
    elif loss_name == "Dice_loss":
        return DiceLoss()
    elif loss_name == "Hybric_loss":
        return HybricLoss()
    elif loss_name == "BCEDice_loss":
        return BCEDiceLoss()
    elif loss_name == "BCEw_loss":
        return BCEWeightLoss(pos_weight=100.0)
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")

# ==========================================
# 5. METRICS & VISUALIZATION (GIỮ NGUYÊN)
# ==========================================

# def dice_coeff_hard(logits, target, threshold=0.5, epsilon=1e-6):
#     probs = torch.sigmoid(logits)
#     preds = (probs > threshold).float()
#     preds_flat = preds.view(preds.size(0), -1)
#     target_flat = target.view(target.size(0), -1)
#     intersection = (preds_flat * target_flat).sum(dim=1)
#     dice = (2. * intersection + epsilon) / (preds_flat.sum(dim=1) + target_flat.sum(dim=1) + epsilon)
#     return dice

# def iou_core_hard(logits, target, threshold=0.5, epsilon=1e-6):
#     probs = torch.sigmoid(logits)
#     preds = (probs > threshold).float()
#     preds_flat = preds.view(preds.size(0), -1)
#     target_flat = target.view(target.size(0), -1)
#     intersection = (preds_flat * target_flat).sum(dim=1)
#     union = preds_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
#     iou = (intersection + epsilon) / (union + epsilon)
#     return iou


def dice_coeff_hard(logits, target, threshold=0.5, epsilon=1e-6):
    """
    Tính Dice Score (Hard Metric).
    Hỗ trợ tự động Deep Supervision (Input là List).
    """
    # --- [THÊM MỚI] XỬ LÝ DEEP SUPERVISION ---
    if isinstance(logits, (list, tuple)):
        # Trong SMP, phần tử đầu tiên (index 0) luôn là output cuối cùng (Final Output)
        logits = logits[0] 
    # -----------------------------------------

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (preds_flat * target_flat).sum(dim=1)
    dice = (2. * intersection + epsilon) / (preds_flat.sum(dim=1) + target_flat.sum(dim=1) + epsilon)
    return dice

def iou_core_hard(logits, target, threshold=0.5, epsilon=1e-6):
    """
    Tính IoU Score (Hard Metric).
    Hỗ trợ tự động Deep Supervision (Input là List).
    """
    # --- [THÊM MỚI] XỬ LÝ DEEP SUPERVISION ---
    if isinstance(logits, (list, tuple)):
        logits = logits[0]
    # -----------------------------------------

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    preds_flat = preds.view(preds.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    
    intersection = (preds_flat * target_flat).sum(dim=1)
    union = preds_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection
    
    iou = (intersection + epsilon) / (union + epsilon)
    return iou
    
def unnormalize(img_tensor):
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array([0.1608, 0.1751, 0.1216])
    std = np.array([0.2526, 0.2466, 0.1983])
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return img

def visualize_prediction(img_tensor, mask_tensor, pred_tensor, save_path, iou_score, dice_score):
    """
    Vẽ ảnh 3 kênh thông minh: Hiển thị kênh CLAHE (Green - Kênh 1) làm background
    để dễ nhìn thấy khối u và đường vân nhất.
    """
    # 1. Khôi phục ảnh 3 kênh đầy đủ
    full_img_rgb = unnormalize(img_tensor) 
    full_img_rgb = full_img_rgb[:, :, ::-1]
    # 2. Chỉ lấy kênh Green (CLAHE) để hiển thị cho rõ (Vì nó sáng và chi tiết nhất)
    # Hoặc hiển thị cả ảnh RGB cũng được, nhưng màu sẽ hơi lạ (tím/xanh).
    # Ở đây mình chọn hiển thị kênh CLAHE (Channel 1) dưới dạng ảnh xám cho chuyên nghiệp.
    # display_img = full_img_rgb[:, :, 1] 
    # orig_img = unnormalize(img_tensor) 
    gt_mask = mask_tensor.squeeze().cpu().numpy()
    pred_mask = pred_tensor.squeeze().cpu().numpy()

    cmap_gt = ListedColormap(['#006400']) 
    cmap_pred = ListedColormap(['#8B0000']) 

    plt.figure(figsize=(12, 4))
    
    # 1. Ảnh gốc (Hiển thị kênh CLAHE)
    plt.subplot(1, 3, 1)
    # plt.imshow(orig_img)
    # plt.imshow(display_img, cmap='gray')
    # plt.imshow(full_img_rgb)
    
    plt.imshow(full_img_rgb)
    plt.title("Input (CLAHE Channel)")
    plt.axis('off')

    # 2. GT
    plt.subplot(1, 3, 2)
    # plt.imshow(display_img, cmap='gray')
    # plt.imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap=cmap_gt, alpha=0.6, interpolation='none')
    plt.imshow(gt_mask, cmap='gray')
    plt.title("Ground Truth")
    plt.axis('off')

    # 3. Overlay
    plt.subplot(1, 3, 3)
    # plt.imshow(display_img, cmap='gray')
    plt.imshow(full_img_rgb)
    # Doi 0.6 thanh 0.4
    plt.imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap=cmap_gt, alpha=0.6, interpolation='none')
    #  Doi 0.4 thanh 0.6
    plt.imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap=cmap_pred, alpha=0.4, interpolation='none')
    plt.title(f"IoU: {iou_score:.2f} | Dice: {dice_score:.2f}")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

