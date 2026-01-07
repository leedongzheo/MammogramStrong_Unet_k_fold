from config import *
from utils import *
from optimizer import *

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, patience=20, device=DEVICE):
        self.device = device
        self.model = model.to(self.device)
        self.num_epochs = NUM_EPOCHS
        self.criterion = criterion
        self.patience = patience
        self.optimizer = optimizer
        self.scheduler = scheduler
        # Tracking metrics
        self.early_stop_counter = 0
        self.reset_count = 0
        # Lưu history riêng biệt để vẽ biểu đồ sau này nếu cần
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_dice_mass': [], 'val_dice_mass': [],
            'train_dice_norm': [], 'val_dice_norm': [],
            'train_iou_mass': [],  'val_iou_mass': [],
            'train_iou_norm': [],  'val_iou_norm': []
        }
        
        # Best metrics tracking
        self.best_dice_mass, self.best_epoch_dice = 0.0, 0
        self.best_iou_mass, self.best_epoch_iou = 0.0, 0
        self.best_epoch_loss = 0
        # --- THÊM: Theo dõi Best Val Loss cho Early Stopping ---
        self.best_val_loss = float('inf')         
        self.log_interval = 1

        # AMP & Scheduler
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler(enabled=self.use_amp)
        # Start epoch
        self.start_epoch = 0

    def save_checkpoint(self, epoch, dice, iou, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict(),   
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None, 
            'scaler_state_dict': self.scaler.state_dict(),
            'history': self.history,
            'best_dice_mass': self.best_dice_mass,
            'best_iou_mass': self.best_iou_mass,
            'best_epoch_dice': self.best_epoch_dice,
            'best_epoch_iou': self.best_epoch_iou,
            # --- THÊM: Lưu best_val_loss ---
            'best_val_loss': self.best_val_loss,
            'best_epoch_loss': self.best_epoch_loss
        }
        torch.save(checkpoint, filename)

    def load_checkpoint(self, path):
        print(f"[INFO] Loading checkpoint: {path}")
        # Thêm weights_only=False
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler and checkpoint['scheduler_state_dict']:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                print(f"[WARNING] Scheduler structure changed (Linear -> Sequential). Resetting Scheduler.")
                print(f"Details: {e}")
                # Không load state cũ, để scheduler chạy lại từ đầu (tự tính toán dựa trên epoch hiện tại)
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.start_epoch = checkpoint['epoch'] 
        
        # Load history
        self.history = checkpoint.get('history', self.history)
        # self.val_ious = checkpoint.get('val_ious', [])
        
        self.best_dice_mass = checkpoint.get('best_dice_mass', 0.0)
        self.best_iou_mass = checkpoint.get('best_iou_mass', 0.0)
        
        # --- THÊM: Load best_val_loss ---
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_epoch_loss = checkpoint.get('best_epoch_loss', 0)
        self.best_epoch_dice = checkpoint.get('best_epoch_dice', 0)
        self.best_epoch_iou = checkpoint.get('best_epoch_iou', 0)
        
        print(f"[INFO] Loaded checkpoint from epoch {self.start_epoch}")

    def run_epoch(self, loader, is_train=True):
        """Hàm chung để chạy train hoặc validation cho 1 epoch"""
        self.model.train() if is_train else self.model.eval()
        
        epoch_loss = 0.0
        total_dice_mass, total_iou_mass = 0.0, 0.0
        count_mass = 0
        
        total_dice_norm, total_iou_norm = 0.0, 0.0
        count_norm = 0
        
        desc = "Training" if is_train else "Validation"
        loader_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=False)
        
        for i, (images, masks, _) in loader_bar:
            images, masks = images.to(self.device), masks.to(self.device)
            
            if is_train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                loss = torch.mean(loss)
                
                with torch.no_grad():
                    # 1. Tính Metric Hard cho từng ảnh trong batch (Tensor [B])
                    # Truyền thẳng logits, hàm utils sẽ tự sigmoid -> threshold
                    batch_dices = dice_coeff_hard(outputs, masks)
                    batch_ious  = iou_core_hard(outputs, masks)
                    # 2. Phân loại Mass vs Normal dựa trên Ground Truth Mask
                    masks_flat = masks.view(masks.size(0), -1)
                    mask_sums = masks_flat.sum(dim=1)
                    is_mass = (mask_sums > 0)  # Có u
                    is_norm = (mask_sums == 0) # Không u (Normal)
                    # 3. Cộng dồn riêng
                    if is_mass.any():
                        total_dice_mass += batch_dices[is_mass].sum().item()
                        total_iou_mass  += batch_ious[is_mass].sum().item()
                        count_mass += is_mass.sum().item()
                    
                    if is_norm.any():
                        total_dice_norm += batch_dices[is_norm].sum().item()
                        total_iou_norm  += batch_ious[is_norm].sum().item()
                        count_norm += is_norm.sum().item()

            if is_train:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # Step Scheduler theo batch (CosineAnnealingWarmRestarts cần điều này)
                # if self.scheduler:
                #     self.scheduler.step(self.current_epoch + i / len(loader)) 

            epoch_loss += loss.item()
            # Hiển thị progress bar
            curr_d_mass = total_dice_mass / count_mass if count_mass > 0 else 0.0
            curr_d_norm = total_dice_norm / count_norm if count_norm > 0 else 0.0
            curr_i_mass = total_iou_mass / count_mass if count_mass > 0 else 0.0
            curr_i_norm = total_iou_norm / count_norm if count_norm > 0 else 0.0
            
            if (i + 1) % self.log_interval == 0:
                loader_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}", 
                    'D_Mass': f"{curr_d_mass:.3f}", 
                    'D_Norm': f"{curr_d_norm:.3f}",
                    'I_Mass': f"{curr_i_mass:.3f}", 
                    'I_Norm': f"{curr_i_norm:.3f}",
                })
        
        avg_loss = epoch_loss / len(loader)
        # Metric cuối cùng của epoch
        final_dice_mass = total_dice_mass / count_mass if count_mass > 0 else 0.0
        final_iou_mass  = total_iou_mass / count_mass if count_mass > 0 else 0.0
        
        final_dice_norm = total_dice_norm / count_norm if count_norm > 0 else 0.0
        final_iou_norm  = total_iou_norm / count_norm if count_norm > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'dice_mass': final_dice_mass, 'iou_mass': final_iou_mass,
            'dice_norm': final_dice_norm, 'iou_norm': final_iou_norm
        }

    def train(self, train_loader, val_loader, resume_path=None):
        print("-" * 30)
        print(f"Device: {self.device}")
        print(f"Num Epochs: {self.num_epochs}")
        print(f"Early Stopping Monitor: Val Loss (Patience={self.patience})")
        print(f"Best Model Monitor: Val IoU & IoU Mass (Hard Metric)")
        print("-" * 30)

        if resume_path:
            self.load_checkpoint(resume_path)
        
        start_time = time.time()
        print(f"[INFO] Starting training from epoch {self.start_epoch + 1}...")
        MIN_LR_THRESHOLD = 1e-6  
        RESET_LR_VALUE = 1e-4
        MIN_LR = 1e-6
        MAX_RESETS = 3
        for epoch in range(self.start_epoch, self.num_epochs):
            self.current_epoch = epoch 
            
            # --- Training ---
            train_res = self.run_epoch(train_loader, is_train=True)
            # --- [THÊM MỚI] Cập nhật Scheduler tại cuối mỗi Epoch ---
            # if self.scheduler:
            #     self.scheduler.step()  # <--- Gọi không tham số
            # --- Validation ---
            with torch.no_grad():
                val_res = self.run_epoch(val_loader, is_train=False)
            # --- [LOGIC SCHEDULER THÔNG MINH] ---
            if self.scheduler:
                # Kiểm tra nếu là ReduceLROnPlateau -> Cần truyền metric
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    # Truyền Dice Mass vào để theo dõi (Cần set mode='max' ở main)
                    self.scheduler.step(val_res['dice_mass'])
                else:
                    # Các Scheduler khác (Cosine, Linear, Arithmetic...) -> Không truyền gì
                    self.scheduler.step()
            # ------------------------------------
            # --- Logging ---
            try:
                current_lr = self.scheduler.get_last_lr()[0]
            except:
                current_lr = self.optimizer.param_groups[0]['lr']
            if not hasattr(self, 'reset_count'): self.reset_count = 0
            # Kiểm tra: Đang dùng ReduceLR VÀ LR đã chạm đáy
            if self.scheduler and isinstance(self.scheduler, ReduceLROnPlateau):
                if current_lr <= MIN_LR_THRESHOLD + 1e-9:
                    if self.reset_count < MAX_RESETS:
                        print(f"\n[CYCLIC STRATEGY] 📉 LR hit bottom ({current_lr:.2e})! Resetting cycle...")
                        # A. Reset LR trong Optimizer lên lại đỉnh (1e-4)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = RESET_LR_VALUE
                        print(f"[CYCLIC STRATEGY] 🚀 Learning Rate reset to {RESET_LR_VALUE}")
                        old_patience = self.scheduler.patience
                        old_factor = self.scheduler.factor
                        old_mode = self.scheduler.mode # 'max'
                        # B. Reset Scheduler (Tạo mới lại chính nó)
                        # Lý do: Để reset bộ đếm patience và best metric về trạng thái ban đầu
                        # Lưu ý: Các tham số này phải khớp với cấu hình trong main()
                        self.scheduler = ReduceLROnPlateau(
                            self.optimizer, 
                            mode = old_mode, 
                            factor = old_factor,   # Dùng lại factor cũ (0.5)
                            patience = old_patience, # Dùng lại patience cũ (10)
                            # verbose = True,
                            min_lr = MIN_LR
                        )
                        print(f"[CYCLIC STRATEGY] 🔄 Scheduler re-initialized! Starting new reduction cycle.")
                        self.early_stop_counter = 0 
                        print(f"[CYCLIC STRATEGY] 🛡️ Early Stopping counter reset to 0!")
                        self.reset_count += 1
                    else:
                        print(f"[CYCLIC STRATEGY] 🛑 Max resets reached ({MAX_RESETS}). No more restarts allowed.")
            # current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{self.num_epochs} | LR: {current_lr:.2e}")
            # In kết quả chi tiết
            print(f"Train - Loss: {train_res['loss']:.4f}")
            print(f"      - Mass: Dice {train_res['dice_mass']:.4f} | IoU {train_res['iou_mass']:.4f}")
            print(f"      - Norm: Dice {train_res['dice_norm']:.4f} | IoU {train_res['iou_norm']:.4f}")
            
            print(f"Val   - Loss: {val_res['loss']:.4f}")
            print(f"      - Mass: Dice {val_res['dice_mass']:.4f} | IoU {val_res['iou_mass']:.4f}")
            print(f"      - Norm: Dice {val_res['dice_norm']:.4f} | IoU {val_res['iou_norm']:.4f}")

            # Lưu history
            # Lưu vào history
            self.history['train_loss'].append(train_res['loss'])
            self.history['val_loss'].append(val_res['loss'])
            self.history['train_dice_mass'].append(train_res['dice_mass'])
            self.history['val_dice_mass'].append(val_res['dice_mass'])
            self.history['train_iou_mass'].append(train_res['iou_mass'])
            self.history['val_iou_mass'].append(val_res['iou_mass'])
            # --- THÊM 4 DÒNG NÀY (BẮT BUỘC) ---
            self.history['train_dice_norm'].append(train_res['dice_norm'])
            self.history['val_dice_norm'].append(val_res['dice_norm'])
            
            self.history['train_iou_norm'].append(train_res['iou_norm'])
            self.history['val_iou_norm'].append(val_res['iou_norm'])
            # --- Checkpoint & Logic tách biệt ---
            
            # 1. Luôn lưu model mới nhất
            self.save_checkpoint(epoch + 1, self.best_dice_mass, self.best_iou_mass, 'last_model.pth')
            if val_res['dice_mass'] > self.best_dice_mass:
                self.best_dice_mass = val_res['dice_mass']
                self.best_epoch_dice = epoch + 1
                self.save_checkpoint(epoch + 1, self.best_dice_mass, self.best_iou_mass, 'best_dice_mass_model.pth')
                print(f"[*] New best Dice: {self.best_dice_mass:.4f} at epoch {epoch+1}")
            # 2. Lưu BEST MODEL dựa trên IoU (Theo yêu cầu)
            if val_res['iou_mass'] > self.best_iou_mass:
                self.best_iou_mass = val_res['iou_mass']
                self.best_epoch_iou = epoch + 1
                
                self.save_checkpoint(epoch + 1, self.best_dice_mass, self.best_iou_mass, 'best_iou_mass_model.pth')
                print(f"[*] New best IoU: {self.best_iou_mass:.4f} at epoch {epoch+1}")

            # 3. EARLY STOPPING dựa trên Val Loss (Theo yêu cầu)
            # Lưu ý: Scheduler ReduceLR cũng có patience riêng (để giảm LR), 
            # còn ở đây là patience để DỪNG TRAIN. Hai cái này hoạt động độc lập.
            if val_res['loss'] < self.best_val_loss:
                self.best_val_loss = val_res['loss']
                self.best_epoch_loss = epoch + 1
                self.early_stop_counter = 0 # Reset counter vì loss giảm (tốt lên)
                # print(f"[*] Best Loss updated: {self.best_val_loss:.4f}") 
            else:
                self.early_stop_counter += 1
                print(f"[!] Loss didn't improve. EarlyStopping counter: {self.early_stop_counter}/{self.patience}")

            if self.early_stop_counter >= self.patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch + 1} due to no improvement in Loss.")
                break
            
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()

        total_time = time.time() - start_time
        print(f"[INFO] Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    def evaluate(self, test_loader, checkpoint_path=None, save_visuals=False, output_dir="test_results"):
        """
        Đã cập nhật để hỗ trợ xuất ảnh dự đoán
        """
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        self.dice_list, self.iou_list, self.path_list, self.type_list = [], [], [], []
        # Biến tích lũy cho test
        total_dice_mass, count_mass = 0.0, 0
        total_dice_norm, count_norm = 0.0, 0
        # Tạo thư mục lưu ảnh nếu cần
        if save_visuals:
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Saving visualization results to: {output_dir}")

        with torch.no_grad():
            test_bar = tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing")
            for i, (images, masks, image_paths) in test_bar:
                images, masks = images.to(self.device), masks.to(self.device)
                # Forward pass
                # để tính probability cho visualization.
                logits = self.model(images)
                if isinstance(logits, (list, tuple)):
                    logits_for_pred = logits[0]
                else:
                    logits_for_pred = logits
                # 1. Tính Metric (Truyền logits thẳng vào, hàm hard tự lo phần còn lại)
                batch_dices = dice_coeff_hard(logits_for_pred, masks, threshold=0.3)
                batch_ious = iou_core_hard(logits_for_pred, masks, threshold=0.3)
                if save_visuals:
                    # Tính xác suất để visualize (0 -> 1)
                    probs = torch.sigmoid(logits_for_pred) # Sigmoid chỉ chạy trên Tensor, không chạy trên List
                    # Tạo mask nhị phân (0 hoặc 1) để vẽ
                    preds = (probs > 0.3).float()
                # Lặp từng ảnh trong batch để tính metric và vẽ
                for j in range(images.size(0)):
                    d = batch_dices[j].item()
                    ious = batch_ious[j].item()
                    path = image_paths[j]
                    # Logic phân loại Mass/Normal
                    is_normal = (masks[j].sum() == 0)
                    current_type = "Normal" if is_normal else "Mass"
                    
                    self.dice_list.append(d)
                    self.iou_list.append(ious)
                    self.path_list.append(path)
                    self.type_list.append(current_type) # <--- QUAN TRỌNG: Để ở đây mới đúng
                    # Logic tách metric cho Test Report
                    if is_normal:
                        total_dice_norm += d
                        count_norm += 1
                    else:
                        total_dice_mass += d
                        count_mass += 1
                    # --- PHẦN BỔ SUNG: VẼ ẢNH ---
                    if save_visuals:
                        # Lấy tên file gốc
                        file_name = os.path.basename(path)
                        # Prefix NORM/MASS
                        prefix = "NORM" if is_normal else "MASS" # Dùng lại biến is_normal ở trên
                        save_name = f"pred_{prefix}_D{d:.2f}_{file_name}"
                        save_full_path = os.path.join(output_dir, save_name)
                        visualize_prediction(
                            img_tensor=images[j],
                            mask_tensor=masks[j],
                            pred_tensor=preds[j], # Dùng preds đã tính ở trên
                            save_path=save_full_path,
                            iou_score=ious,
                            dice_score=d
                        )
                    # -----------------------------

        # Báo cáo kết quả tách biệt
        avg_dice_mass = total_dice_mass / count_mass if count_mass > 0 else 0.0
        avg_dice_norm = total_dice_norm / count_norm if count_norm > 0 else 0.0
        # print(f"\n[TEST RESULT] Avg Hard Dice: {avg_dice:.4f}, Avg Hard IoU: {avg_iou:.4f}")
        print(f"\n[TEST REPORT]")
        print(f"   - Mass Samples: {count_mass} | Avg Dice: {avg_dice_mass:.4f}")
        print(f"   - Norm Samples: {count_norm} | Avg Dice: {avg_dice_norm:.4f}") # Chỉ số này nên là 1.0 hoặc gần 1.0
        return avg_dice_mass, avg_dice_norm, self.dice_list, self.iou_list, self.path_list

    def get_metrics(self):
        return {
            'history': self.history, 
            # 'best_dice': self.best_dice,
            'best_dice_mass': self.best_dice_mass,
            'best_epoch_dice': self.best_epoch_dice,
            # 'best_iou': self.best_iou,
            'best_iou_mass': self.best_iou_mass,
            'best_epoch_iou': self.best_epoch_iou,
            'best_val_loss': self.best_val_loss,
            'best_epoch_loss': self.best_epoch_loss,
        }
