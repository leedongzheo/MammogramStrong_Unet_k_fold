from config import*
# ==============================================================================
# 1. CẤU HÌNH LOGIC AUGMENTATION (DYNAMIC)
# ==============================================================================
# Các tham số xác suất (giống file offline)
AUG_PROBS = {
    'trigger_spatial': 0.8,
    'trigger_pixel':   0.5,
    'spatial_hflip':   0.5,
    'spatial_rotate':  0.5,
    # 'spatial_deform':  0.6,
    'spatial_deform':  0.4
}

# Đường dẫn đến file metadata (đã tạo bằng script offline)
# METADATA_PATH = "train_metadata.csv"
# Giá trị an toàn mặc định nếu không tìm thấy trong CSV
GLOBAL_MIN_AREA_DEFAULT = 143.5  
# Mean và Std chuẩn cho bộ 3 kênh (Original, CLAHE, Gamma)
# NORM_MEAN = [0.1608, 0.1751, 0.1216]
# NORM_STD  = [0.2526, 0.2466, 0.1983]
# ImageNet:
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
# --- HELPER FUNCTIONS ---
def get_clean_breast_mask(image):
    if len(image.shape) == 3:
        gray = image[:, :, 1]
    else:
        gray = image.copy()
    _, binary = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_clean = np.zeros_like(gray)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask_clean, [c], -1, 255, thickness=cv2.FILLED)
    return mask_clean

def add_targeted_coarse_dropout(image, num_holes, min_h, max_h, min_w, max_w):
    h, w = image.shape[:2]
    image_aug = image.copy()
    clean_mask = get_clean_breast_mask(image)
    tissue_coords = np.argwhere(clean_mask > 0)
    
    if len(tissue_coords) == 0: return image_aug
        
    for _ in range(num_holes):
        idx = random.randint(0, len(tissue_coords) - 1)
        cy, cx = tissue_coords[idx]
        hh, ww = random.randint(min_h, max_h), random.randint(min_w, max_w)
        y1, x1 = max(0, cy - hh//2), max(0, cx - ww//2)
        y2, x2 = min(h, y1 + hh), min(w, x1 + ww)
        image_aug[y1:y2, x1:x2] = 0
    return image_aug

# ==============================================================================
# 2. CUSTOM AUGMENTATION CLASS (ĐÃ FIX ĐỘ TRỄ DIỆN TÍCH)
# ==============================================================================
class OnlineStrongAugmentation:
    def __init__(self, output_size=640, mean=None, std=None):
        self.output_size = output_size
        # Xử lý logic: Nếu truyền mean/std riêng thì dùng, không thì dùng mặc định (toàn cục)
        self.mean = mean if mean is not None else NORM_MEAN
        self.std  = std  if std  is not None else NORM_STD
        # self.global_min_area = global_min_area
        
        self.resize = A.Resize(height=output_size, width=output_size, interpolation=cv2.INTER_LINEAR)
        
        self.normalize_and_tensor = A.Compose([
            A.Normalize(mean=self.mean, std=self.std), 
            ToTensorV2()
        ])

    # --- HÀM MỚI: TÍNH DIỆN TÍCH TỨC THỜI ---
    def _get_current_area(self, mask):
        """
		Tính diện tích mask hiện tại bằng Contour (Chính xác hơn countNonZero).
		Giúp loại bỏ các mảnh vỡ/nhiễu nhỏ do phép biến đổi hình học gây ra.
		"""
        # --- BƯỚC 0: VỆ SINH DỮ LIỆU (QUAN TRỌNG) ---
        if mask.dtype != np.uint8:
            # Nếu mask là float trong khoảng [0, 1] (thường gặp)
            if mask.max() <= 1.5: 
                mask = (mask * 255).astype(np.uint8)
            else:
                # Nếu mask là float [0, 255]
                mask = mask.astype(np.uint8)
		# 1. Threshold để nhị phân hóa
        _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
		
		# 2. Tìm contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		
		# 3. Nếu không tìm thấy contour nào (ảnh đen xì) -> Trả về 0
        if not contours:
            return 0.0
			
		# 4. Lấy contour có diện tích lớn nhất (Bỏ qua các đốm nhiễu nhỏ)
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
		
        return area # Trả về số thực (float), ví dụ 195.5

    def apply_spatial(self, image, mask, mass_area, min_dataset_area):
        """
        min_dataset_area: Diện tích nhỏ nhất tìm thấy trong toàn bộ file CSV.
        """
        
        # --- CASE 1: ẢNH NORMAL (KHÔNG CÓ U) ---
        # Không cần kiểm tra diện tích, chỉ cần apply augmentation 1 lần
        if mass_area == 0:
            return self._run_spatial_transforms(image, mask, mass_area)

        # --- CASE 2: ẢNH CÓ MASS ---
        # Cần cơ chế Retry để đảm bảo u không bị teo nhỏ quá mức
        max_attempts = 5  # Thử tối đa 5 lần
        
        for _ in range(max_attempts):
            # Chạy thử augmentation
            aug_img, aug_mask = self._run_spatial_transforms(image, mask, mass_area)
            if aug_mask.dtype != np.uint8:
                aug_mask = ((aug_mask > 0) * 255).astype(np.uint8)
            # Tìm contours (chỉ lấy đường bao ngoài - RETR_EXTERNAL)
            contours, _ = cv2.findContours(aug_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                new_area = 0
            else:
                new_area = max(cv2.contourArea(c) for c in contours)
            threshold_area = max(30, min_dataset_area * 0.5)
            # ĐIỀU KIỆN CHẤP NHẬN:
            # Diện tích mới không được nhỏ hơn diện tích nhỏ nhất của dataset
            if new_area >= threshold_area:
                return aug_img, aug_mask
        
        # Nếu thử 5 lần mà lần nào u cũng bị cắt cụt/thu nhỏ quá mức
        # -> Trả về ảnh gốc để bảo toàn dữ liệu
        return image, mask

    def _run_spatial_transforms(self, image, mask, mass_area):
        """
        Hàm phụ: Chứa logic tạo transform list cũ của bạn.
        Tách ra để dễ gọi lại trong vòng lặp.
        """
        transforms_list = []
        
        # Thiết lập tham số Alpha cho Elastic
        if mass_area > 0:
            mass_side = np.sqrt(mass_area)
            alpha = max(30, min(mass_side * 4.0, 80))
        else:
            alpha = random.uniform(30, 100)
            
        sigma = alpha * 0.12

        # 1. Horizontal Flip
        if random.random() < AUG_PROBS['spatial_hflip']:
            transforms_list.append(A.HorizontalFlip(p=1.0))

        # 2. Rotate
        # 2. Shift - Scale - Rotate (Nâng cấp)
        # Kết hợp xoay, phóng to/nhỏ và dịch chuyển nhẹ
        if random.random() < AUG_PROBS['spatial_rotate']: # Tái sử dụng xác suất cũ
            transforms_list.append(A.ShiftScaleRotate(
                # shift_limit=0.0625, # Dịch chuyển tối đa 6.25% (nhẹ nhàng)
				shift_limit=0.1,    # Tăng từ 0.0625 -> 0.1 (Dịch chuyển nhiều hơn)
                # scale_limit=0.15,   # Zoom in/out tối đa 15% (An toàn, không làm vỡ ảnh)
				scale_limit=0.35,    # Tăng từ 0.15 -> 0.3 (Zoom to/nhỏ mạnh hơn)
                # rotate_limit=15,    # Xoay tối đa 15 độ (như cũ)
				rotate_limit=45,    # Tăng từ 15 -> 45 (Xoay nghiêng ngả hơn)
                border_mode=cv2.BORDER_CONSTANT, value=0, 
                p=1.0
            ))

        # 3. Deformation
        if random.random() < AUG_PROBS['spatial_deform']:
            choice = random.choice(['Elastic', 'GridDistortion'])
            if choice == 'Elastic':
                transforms_list.append(A.ElasticTransform(
                    alpha=alpha, sigma=sigma, alpha_affine=alpha*0.1, 
                    border_mode=cv2.BORDER_REFLECT_101, p=1.0
                ))
            else:
                transforms_list.append(A.GridDistortion(
                    num_steps=random.randint(5, 8), 
                    distort_limit=0.15, 
                    border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0
                ))

        if not transforms_list:
            return image, mask

        aug = A.Compose(transforms_list)
        transformed = aug(image=image, mask=mask)
        return transformed['image'], transformed['mask']

    # # apply_pixel giữ nguyên như bạn yêu cầu
    # def apply_pixel(self, image, current_area):
    #     # ... (Code apply_pixel cũ của bạn) ...
    #     pass

    def apply_pixel(self, image, current_area, min_dataset_area):
        # 1. TÍNH TOÁN safe_side_limit (KÍCH THƯỚC LỖ TỐI ĐA)
        # 1. LỌC DANH SÁCH AN TOÀN (Giữ nguyên logic cũ)
        valid_choices = ['GaussNoise']
        # Tính margin: current=5782, min=143 -> margin = 40 lần (Rất an toàn)
        safety_margin = current_area / min_dataset_area if min_dataset_area > 0 else 999
        if safety_margin > 1.2:
            valid_choices.extend(['GridDropout', 'CoarseDropout'])
        choice = random.choice(valid_choices)
        image_aug = image.copy()
        # pixel_params = "" # Để debug
        if current_area > 0:
            # --- TRƯỜNG HỢP CÓ KHỐI U (MASS) ---
            # Phải tính toán cẩn thận để không xóa mất khối u
            # Công thức: max(8, min(sqrt(0.4 * Area), 30))
            # Kết quả: U nhỏ -> Lỗ 8px. U to -> Lỗ 30px.
            # safe_side_limit = max(8, min(int(np.sqrt(0.4 * current_area)), 30))
            safe_side_limit = max(12, min(int(np.sqrt(0.4 * current_area)), 50))            
        else:
            # --- TRƯỜNG HỢP NORMAL (KHÔNG CÓ U) ---
            # Không sợ che mất u, nên cho phép lỗ to hơn để tăng độ khó (Strong Aug)
            # Random cận trên từ 15 đến 30 pixel (thay vì cố định 8px như cũ)
            # safe_side_limit = random.randint(15, 30)
            safe_side_limit = random.randint(25, 50)

        # choice = random.choice(['GridDropout', 'CoarseDropout', 'GaussNoise', 'BrightnessContrast'])
        # image_aug = image.copy()
        target_hole_size = random.randint(max(6, safe_side_limit - 2), safe_side_limit)
        if choice == 'GridDropout':
            # --- BƯỚC 1: XÁC ĐỊNH CHIẾN THUẬT (STRATEGY) ---
            # Thay vì random bừa, ta chọn chiến thuật rõ ràng
            
            # Tính Max Safe Ratio (Chốt chặn an toàn)
            max_safe_ratio = 1.0 - (min_dataset_area / current_area) if current_area > 0 else 0.5
            
            if safety_margin > 5.0: 
                # === KHỐI U LỚN ===
                if random.random() < 0.5:
                    # Chiến thuật 1: Cấu trúc (Lỗ TO - Ratio THẤP)
                    # Nới trần lên 60px thay vì 30px
                    # real_limit = max(31, min(int(np.sqrt(0.4 * current_area)), 60))
                    real_limit = max(50, min(int(np.sqrt(0.4 * current_area)), 100))
                    target_hole_size = random.randint(50, real_limit)
                    # target_hole_size = random.randint(30, real_limit) 
                    suggested_ratio_min, suggested_ratio_max = 0.15, 0.3
                else:
                    # Chiến thuật 2: Texture (Lỗ NHỎ - Ratio CAO)
                    target_hole_size = random.randint(10, 20)
                    suggested_ratio_min, suggested_ratio_max = 0.35, 0.5
            else:
                # === KHỐI U VỪA & NHỎ ===
                # Lỗ nhỏ vừa phải, Ratio trung bình
                # Không dùng ratio quá thấp (0.1) vì vô dụng
                # target_hole_size = random.randint(6, 12)
                suggested_ratio_min, suggested_ratio_max = 0.25, 0.4

            # --- BƯỚC 2: TÍNH TOÁN ---
            
            # Kẹp ratio theo chiến thuật, NHƯNG không được vượt quá max_safe_ratio
            final_upper = min(suggested_ratio_max, max_safe_ratio, 0.5)
            final_lower = min(suggested_ratio_min, final_upper)
            
            if final_upper < final_lower: 
                return image_aug # Fallback

            ratio = random.uniform(final_lower, final_upper)
            
            # Tính Unit Size bù trừ
            calculated_unit_size = int(target_hole_size / ratio)
            u_size = max(2, calculated_unit_size)
            
            aug = A.GridDropout(ratio=ratio, unit_size_min=u_size, unit_size_max=u_size, fill_value=0, p=1.0)
            image_aug = aug(image=image)['image']
            
        elif choice == 'CoarseDropout':
            # U to thì nhiều lỗ (hoặc lỗ to), U nhỏ thì ít lỗ
            if safety_margin > 5.0:
                # === KHỐI U LỚN ===
                # Kiểm tra xem lỗ to hay nhỏ để quyết định số lượng
                if target_hole_size > 20: 
                    # Lỗ to (>20px) -> Chỉ cho phép 2-4 lỗ (Tránh xóa sạch u)
                    actual_holes = random.randint(2, 4)
                else:
                    # Lỗ nhỏ (<=20px) -> Cho phép 5-10 lỗ (Tạo hiệu ứng mưa rào/lưới)
                    actual_holes = random.randint(5, 10)
                    
            elif safety_margin > 2.0:
                # === KHỐI U VỪA ===
                if target_hole_size > 15:
                    actual_holes = random.randint(1, 3)
                else:
                    actual_holes = random.randint(3, 5)
            else:
                # === KHỐI U NHỎ (SÁT VÁN) ===
                # An toàn tuyệt đối: Luôn chỉ 1 lỗ
                actual_holes = 1
            # actual_holes = random.randint(4, 8)
            h_min = max(2, target_hole_size - 2)
            h_max = target_hole_size + 2
            
            # 2. Tính toán phạm vi cho Chiều Rộng (Width)
            # Logic: Width được phép random từ "Một nửa Height" đến "Gấp rưỡi Height"
            # Nhưng việc chọn số cụ thể là ngẫu nhiên hoàn toàn.
            w_min = max(2, int(target_hole_size * 0.5)) 
            w_max = max(w_min + 1, int(target_hole_size * 1.5))

            # 3. Gọi hàm đục lỗ
            # Hàm này sẽ bốc h từ [h_min, h_max] và w từ [w_min, w_max] riêng biệt
            image_aug = add_targeted_coarse_dropout(
                image, actual_holes, 
                min_h=h_min, max_h=h_max, 
                min_w=w_min, max_w=w_max
            )
            
        elif choice == 'GaussNoise':
            aug = A.GaussNoise(var_limit=(10.0, 25.0), p=1.0)
            image_aug = aug(image=image)['image']

        return image_aug

    def __call__(self, image, mask, mass_area=0.0, min_dataset_area=0.0):
        # 1. Resize trước tiên (Bắt buộc)
        # Lưu ý: Resize cũng làm thay đổi diện tích khối u so với CSV.
        resized = self.resize(image=image, mask=mask)
        img_aug, mask_aug = resized['image'], resized['mask']
        
        # 2. Trigger Spatial Augmentation
        # Ta vẫn dùng mass_area (CSV) để tính tham số alpha/sigma cho Elastic 
        # vì nó mang tính tương đối, sai số do resize chấp nhận được ở bước này.
        if random.random() < AUG_PROBS['trigger_spatial']:
            img_aug, mask_aug = self.apply_spatial(
                image=img_aug, 
                mask=mask_aug, 
                mass_area=mass_area,             # Diện tích gốc từ CSV (để tính alpha/sigma Elastic)
                min_dataset_area=min_dataset_area # Ngưỡng diện tích an toàn (để kiểm tra retry)
            )
            
        # 3. [BƯỚC MỚI] Cập nhật lại diện tích thực tế
        # Lúc này img_aug đã bị Resize + Xoay + Méo. Diện tích pixel đã thay đổi.
        # Chúng ta cần con số chính xác này để đục lỗ (Pixel Aug) cho an toàn.
        
        # Nếu trigger_pixel bật thì mới tốn công tính toán, không thì thôi cho nhanh
        if random.random() < AUG_PROBS['trigger_pixel']:
            # Tính diện tích mới ngay lập tức
            current_real_area = self._get_current_area(mask_aug)
            
            # Truyền diện tích mới vào hàm apply_pixel
            img_aug = self.apply_pixel(img_aug, current_real_area, min_dataset_area)

        # 4. Normalize & ToTensor
        final = self.normalize_and_tensor(image=img_aug, mask=mask_aug)
        return final

# ==============================================================================
# 3. DATASET CLASS (LOAD CSV & XỬ LÝ)
# ==============================================================================
class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms, metadata_path=None):
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
        self.min_dataset_area = 0.0
        # --- LOAD METADATA ---
        self.area_lookup = {}
        if metadata_path and os.path.exists(metadata_path):
            try:
                print(f"[DATASET] Loading augmentation metadata from {metadata_path}...")
                df = pd.read_csv(metadata_path)
                # Chuyển thành dict {filename: area} để tra cứu O(1)
                self.area_lookup = df.set_index('filename')['mass_area'].to_dict()
                mass_only = df[df['mass_area'] > 0]
                if not mass_only.empty:
                    self.min_dataset_area = mass_only['mass_area'].min()
                else:
                    self.min_dataset_area = 0.0 # Fallback nếu dataset toàn normal
                print(f"[DATASET] Metadata loaded successfully ({len(self.area_lookup)} records). Min Mass Area in CSV: {self.min_dataset_area}")
            except Exception as e:
                print(f"[WARNING] Could not load metadata: {e}")
        
    def __len__(self):
        return len(self.imagePaths)

    def __getitem__(self, idx):
        imagePath = self.imagePaths[idx]
        maskPath = self.maskPaths[idx]
        
        # Lấy diện tích khối u từ metadata (nếu có), mặc định là 0 (Normal)
        filename = os.path.basename(maskPath)
        mass_area = self.area_lookup.get(filename, 0.0)

        # Đọc ảnh và mask
        image = cv2.imread(imagePath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE) # 0-255

        # Áp dụng Transform
        if self.transforms:
            # Kiểm tra xem transform là loại nào
            if isinstance(self.transforms, OnlineStrongAugmentation):
                # Nếu là Custom Augmentation -> Truyền thêm mass_area
                augmented = self.transforms(
                    image=image, 
                    mask=mask, 
                    mass_area=mass_area, 
                    min_dataset_area=self.min_dataset_area # <--- THAM SỐ MỚI
                )
            else:
                # Nếu là Albumentations thường (Valid/Weak) -> Không truyền mass_area
                augmented = self.transforms(image=image, mask=mask)
            
            image = augmented["image"]
            mask = augmented["mask"]
            
            # Xử lý mask sau khi augment (đảm bảo là Tensor float 0-1)
            if isinstance(mask, torch.Tensor):
                mask = (mask > 0.5).float()
                # ToTensorV2 output shape thường là (H, W) nếu mask input là (H, W)
                # Cần unsqueeze lên (1, H, W)
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0)
            else:
                # Fallback nếu transform không có ToTensorV2 (trường hợp hiếm)
                mask = (mask > 127).astype("float32")
                mask = torch.from_numpy(mask)
                mask = mask.unsqueeze(0)

        return image, mask, imagePath

def seed_worker(worker_id):
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)
class EnsembleModel(nn.Module):
    def __init__(self, model_class, checkpoint_paths, device='cuda', in_channels=3, num_classes=1):
        super().__init__()
        self.models = nn.ModuleList()
        self.device = device
        
        print(f"[ENSEMBLE] Loading {len(checkpoint_paths)} models...")
        
        for path in checkpoint_paths:
            # 1. Khởi tạo kiến trúc model (VD: UNet, ResUnet...)
            # Lưu ý: model_class là tên Class model gốc của bạn (ví dụ: UNet)
            model = model_class(in_channels=in_channels, num_classes=num_classes)
            
            # 2. Load weights
            checkpoint = torch.load(path, map_location=device)
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                 model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # 3. Eval mode & Freeze
            model.to(device)
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
                
            self.models.append(model)
            print(f"  + Loaded: {path}")

    def forward(self, x):
        outputs = []
        # 1. Lấy xác suất từ từng model
        with torch.no_grad():
            for model in self.models:
                logits = model(x)
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                
                # Chuyển Logits -> Probabilities (0-1) để cộng gộp
                probs = torch.sigmoid(logits)
                outputs.append(probs)
        
        # 2. Tính trung bình (Soft Voting)
        avg_probs = torch.stack(outputs).mean(dim=0)
        
        # 3. TRICK QUAN TRỌNG: Chuyển ngược về Logits (Inverse Sigmoid)
        # Vì hàm evaluate() của bạn có dòng `probs = torch.sigmoid(logits)`
        # Nên ta phải trả về logits để khi evaluate sigmoid lần nữa nó ra đúng avg_probs ban đầu.
        # Công thức: logit(p) = log(p / (1 - p))
        eps = 1e-7 # Tránh chia cho 0
        avg_probs_clamped = torch.clamp(avg_probs, eps, 1 - eps)
        avg_logits = torch.log(avg_probs_clamped / (1 - avg_probs_clamped))
        
        return avg_logits
# ==============================================================================
# 4. GET DATALOADERS FUNCTION
# ==============================================================================
def get_dataloaders(aug_mode='none', state='train', fold_idx=0):   
    print(f"\n[INFO] Initializing Dataloaders for FOLD {fold_idx} | Augmentation: {aug_mode.upper()}")
    # 0. Lấy Mean/Std và Metadata của Fold hiện tại
    # stats = FOLD_STATS[fold_idx]
    # norm_mean = stats['mean']
    # norm_std = stats['std']
    metadata_csv = FOLD_METADATA[fold_idx]
	norm_mean = NORM_MEAN # [0.485, 0.456, 0.406]
	norm_std  = NORM_STD  # [0.229, 0.224, 0.225]
    # 1. Định nghĩa Transform dựa trên mode
    if aug_mode == 'strong':
        # Dùng Class Custom vừa viết, truyền global_min_area mặc định
        train_transform = OnlineStrongAugmentation(
            output_size=INPUT_IMAGE_WIDTH,
            mean=norm_mean,
            std=norm_std
        )
    elif aug_mode == 'weak':
        print("[INFO] Loading WEAK Augmentation (Flip only)")
        train_transform = A.Compose([
            A.Resize(height=INPUT_IMAGE_WIDTH, width=INPUT_IMAGE_WIDTH, interpolation=cv2.INTER_LINEAR),
            A.HorizontalFlip(p=0.5), 
			# A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=255.0),
            ToTensorV2()
        ])
    else:
        print("[INFO] Not using AUGMENTATION (Resize & Normalize only)")
        train_transform = A.Compose([
            A.Resize(height=INPUT_IMAGE_WIDTH, width=INPUT_IMAGE_WIDTH, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=255.0),
            ToTensorV2()
        ])

    # Valid Transform (Cố định)
    valid_transform = A.Compose([
        A.Resize(height=INPUT_IMAGE_WIDTH, width=INPUT_IMAGE_WIDTH, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=norm_mean, std=norm_std, max_pixel_value=255.0),
        ToTensorV2()
    ])

    # 2. XÂY DỰNG ĐƯỜNG DẪN ĐỘNG THEO FOLD
    # Cấu trúc: INBREAST_5folds/fold_X/train/images
    fold_dir = os.path.join(DATASET_ROOT_4FOLDS, f"fold_{fold_idx}")
    train_img_dir = os.path.join(fold_dir, "train", "images")
    train_mask_dir = os.path.join(fold_dir, "train", "masks")
    valid_img_dir = os.path.join(fold_dir, "valid", "images")
    valid_mask_dir = os.path.join(fold_dir, "valid", "masks")
    # 2. Load Paths (Sử dụng đường dẫn từ config)
    trainImagesPaths = sorted(list(paths.list_images(train_img_dir)))
    trainMasksPaths = sorted(list(paths.list_images(train_mask_dir)))
    validImagesPaths = sorted(list(paths.list_images(valid_img_dir)))
    validMasksPaths = sorted(list(paths.list_images(valid_mask_dir)))
    # testImagesPaths = sorted(list(paths.list_images(IMAGE_TEST_PATH)))
    # testMasksPaths = sorted(list(paths.list_images(MASK_TEST_PATH)))
    # Trong 4-Fold, Valid cũng chính là Test của vòng đó -> Dùng chung đường dẫn
    
    test_img_dir = os.path.join(DATASET_ROOT_4FOLDS, "test", "images")
    test_mask_dir = os.path.join(DATASET_ROOT_4FOLDS, "test", "masks")
    testImagesPaths = sorted(list(paths.list_images(test_img_dir)))
    testMasksPaths = sorted(list(paths.list_images(test_mask_dir)))
    # 3. Khởi tạo Dataset
    # Chỉ truyền metadata cho tập Train để chạy Dynamic Augmentation
    trainDS = SegmentationDataset(
        trainImagesPaths, 
        trainMasksPaths, 
        transforms=train_transform,
        metadata_path=metadata_csv # <--- Dùng file CSV riêng của từng fold
    )
    
    validDS = SegmentationDataset(validImagesPaths, validMasksPaths, transforms=valid_transform)
    testDS = SegmentationDataset(testImagesPaths, testMasksPaths, transforms=valid_transform)

    print(f"[INFO] Found {len(trainDS)} training images")
    print(f"[INFO] Found {len(validDS)} validation images")
    # print(f"[INFO] Found {len(testDS)} test images")
    train_targets = []
    # --- CÁCH MỚI: Đọc từ CSV (Siêu nhanh) ---
    if os.path.exists(metadata_csv):
        print(f"[INFO] Loading targets directly from {metadata_csv} (Fast)...")
        df = pd.read_csv(metadata_csv)
        
        # Tạo từ điển mapping: Tên file -> Nhãn (0 hoặc 1)
        # mass_area > 0 là Mass (1), ngược lại là Normal (0)
        filename_to_label = dict(zip(df['filename'], (df['mass_area'] > 0).astype(int)))
        
        # Duyệt qua list đường dẫn ảnh đã sort để đảm bảo thứ tự khớp với Dataset
        missing_count = 0
        for path in trainMasksPaths:
            fname = os.path.basename(path)
            if fname in filename_to_label:
                train_targets.append(filename_to_label[fname])
            else:
                # Fallback an toàn: Nếu file ảnh có trong folder nhưng ko có trong CSV
                # Mặc định coi là Normal (0) để không lỗi code
                train_targets.append(0) 
                missing_count += 1
                
        if missing_count > 0:
            print(f"[WARNING] Found {missing_count} images not in CSV. Defaulted to Class 0.")
            
    else:
        # --- CÁCH CŨ: Quét từng ảnh (Chậm - Fallback nếu mất file CSV) ---
        print(f"[INFO] Metadata CSV for Fold {fold_idx} not found at {metadata_csv}. Scanning image files (Slow)...")
        for maskPath in tqdm(trainMasksPaths, desc="Scanning for Sampler"):
            mask = cv2.imread(maskPath, cv2.IMREAD_GRAYSCALE)
            train_targets.append(1 if cv2.countNonZero(mask) > 0 else 0)
    
    train_targets = torch.tensor(train_targets)
    class_counts = torch.bincount(train_targets)
    
    # Xử lý trường hợp chỉ có 1 class (hiếm nhưng an toàn)
    num_normal = class_counts[0].item()
    num_mass = class_counts[1].item() if len(class_counts) > 1 else 0
    
    print(f"[INFO] Distribution -> Normal: {num_normal} | Mass: {num_mass}")
    
    # --- LOGIC SAMPLER MỚI ---
    # Chỉ bật Sampler khi đang TRAIN và không phải là EVALUATE
    if state == 'evaluate':
        print("[INFO] Evaluation Mode: Using Sequential Sampler (No Duplicates - Full Dataset)")
        sampler = None
        shuffle = False # Duyệt tuần tự 1474 ảnh
    
    elif state == 'train': # Đang Train có Augment
        print("[INFO] Training Mode: Using WeightedRandomSampler (Balanced Batch)")
        # ... (Code tính weights giữ nguyên) ...
        weight_normal = 1. / num_normal if num_normal > 0 else 0
        weight_mass = 2. / num_mass if num_mass > 0 else 0
        class_weights = torch.tensor([weight_normal, weight_mass])
	    
        samples_weights = class_weights[train_targets]
        sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
        shuffle = False # [FIXED] Có sampler thì BẮT BUỘC tắt shuffle
    # else: # Train không Augment hoặc trường hợp khác
    #     sampler = None
    #     shuffle = True # Shuffle cho ngẫu nhiên nếu không dùng sampler
    # 5. Tạo DataLoaders
    g = torch.Generator()
    g.manual_seed(SEED)

    trainLoader = DataLoader(
        trainDS, 
        batch_size=batch_size, 
        sampler=sampler,        # Dùng sampler để cân bằng batch
        shuffle=shuffle,          # Sampler bật thì Shuffle phải tắt
        pin_memory=PIN_MEMORY,
        num_workers=4, 
        worker_init_fn=seed_worker, 
        generator=g
    )

    validLoader = DataLoader(
        validDS, 
        shuffle=False, 
        batch_size=batch_size*2, 
        pin_memory=PIN_MEMORY,
        num_workers=4, 
        worker_init_fn=seed_worker, 
        generator=g
    )
    
    testLoader = DataLoader(
        testDS, 
        shuffle=False,
        batch_size=batch_size*2, 
        pin_memory=PIN_MEMORY,
        num_workers=4, 
        worker_init_fn=seed_worker, 
        generator=g
    )   
    
    return trainLoader, validLoader, testLoader
