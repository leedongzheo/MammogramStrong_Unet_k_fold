import os
import torch
# import smp
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from config import*
# --- IMPORT CÁC MODULE CỦA BẠN ---
# (Đảm bảo các file trainer.py, model.py, ... nằm cùng thư mục)
from trainer import Trainer
import optimizer as optimizer_module
from dataset import get_dataloaders
from result import export, export_evaluate
from utils import get_loss_instance

# --- CẤU HÌNH ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_OUTPUT = "output"
BATCH_SIZE = 32  # Chỉnh lại cho khớp với VRAM của bạn

def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    print("-" * 50)
    print(f"[INFO] Mode: SWA RECOVERY (Chạy lại riêng SWA)")
    print("-" * 50)
    set_seed()

    # 1. KHỞI TẠO MODEL (Phải giống hệt lúc train)
    print(f"[INFO] Initializing Model...")
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b4",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
        deep_supervision=True,
        encoder_params={"dropout_rate": 0.5}
    )

    # 2. KHỞI TẠO OPTIMIZER & LOSS
    opt = optimizer_module.optimizer(model=model)
    # SWA thường dùng với FocalTversky hoặc loss bạn đã dùng cuối cùng
    criterion_init = get_loss_instance("FocalTversky_loss") 

    # 3. KHỞI TẠO TRAINER
    trainer = Trainer(model=model, optimizer=opt, criterion=criterion_init, patience=10, device=DEVICE)

    # 4. LOAD DỮ LIỆU (Load Strong Augment cho SWA)
    print("[INFO] Loading Dataloaders...")
    trainLoader_strong, validLoader, _ = get_dataloaders(aug_mode='strong', batch_size=BATCH_SIZE)

    # =========================================================
    # GIAI ĐOẠN 4: SWA FINETUNING (CHẠY LẠI)
    # =========================================================
    
    # [QUAN TRỌNG] Trỏ vào file checkpoint tốt nhất bạn đang có
    # Kiểm tra xem file này có tồn tại không
    checkpoint_to_load = "best_dice_mass_model.pth" 
    
    if os.path.exists(checkpoint_to_load):
        print(f"[INFO] Loading checkpoint for SWA start: {checkpoint_to_load}")
        trainer.load_checkpoint(checkpoint_to_load)
    else:
        print(f"[ERROR] Không tìm thấy file {checkpoint_to_load}!")
        print("Hãy đổi tên biến 'checkpoint_to_load' thành đường dẫn file model bạn muốn dùng.")
        return

    print("\n" + "="*40)
    print(" GIAI ĐOẠN 4: SWA FINETUNING (Fixed Version)")
    print(" Strategy: Constant LR | No Early Stop | 20 Epochs")
    print("="*40)

    # 1. Khởi tạo SWA Model
    swa_model = AveragedModel(trainer.model)
    
    # 2. Cấu hình Scheduler SWA
    swa_lr = 5e-5 
    swa_scheduler = SWALR(trainer.optimizer, swa_lr=swa_lr, anneal_epochs=3)
    print(f"[CONFIG] SWA Scheduler set. LR: {swa_lr}")

    # 3. Vòng lặp SWA
    SWA_EPOCHS = 20
    trainer.patience = 999 
    trainer.early_stop_counter = 0
    
    print("[INFO] Starting SWA Loop...")
    
    # 
    
    for epoch in range(SWA_EPOCHS):
        # --- [FIX QUAN TRỌNG TẠI ĐÂY] ---
        trainer.num_epochs = 1 
        trainer.start_epoch = 0 # <--- RESET VỀ 0 ĐỂ VÒNG LẶP CHẠY ĐƯỢC
        # --------------------------------
        
        trainer.scheduler = swa_scheduler
        
        print(f"\n[SWA] Epoch {epoch+1}/{SWA_EPOCHS}")
        # Chạy 1 epoch train (không load checkpoint nữa vì đã load ở trên rồi)
        trainer.train(trainLoader_strong, validLoader, resume_path=None)
        
        # Cập nhật trọng số trung bình
        swa_model.update_parameters(trainer.model)
        swa_scheduler.step()
        
    # 4. Cập nhật Batch Norm (Bước bắt buộc)
    print("\n[INFO] Updating Batch Normalization statistics for SWA Model...")
    update_bn(trainLoader_strong, swa_model, device=DEVICE)

    # 5. Lưu và Đánh giá SWA Model
    swa_save_path = os.path.join(BASE_OUTPUT, "best_model_swa_fixed.pth") # Đổi tên file chút để không ghi đè cái cũ nếu sợ
    print(f"[INFO] Saving SWA Model to {swa_save_path}")
    
    swa_checkpoint = {
        'epoch': SWA_EPOCHS,
        'model_state_dict': swa_model.state_dict(),         
        'optimizer_state_dict': trainer.optimizer.state_dict(),
        'best_val_loss': trainer.best_val_loss, 
        'best_dice_mass': trainer.best_dice_mass,
        'history': trainer.history,
    }
    torch.save(swa_checkpoint, swa_save_path)
    
    # Xuất kết quả (Dùng hàm export an toàn mới sửa)
    export(trainer)
    
    # Đánh giá Model SWA
    print("\n[INFO] Evaluating SWA Model...")
    trainer.model = swa_model
    
    visual_folder = os.path.join(BASE_OUTPUT, "prediction_images_swa_fixed")
    os.makedirs(visual_folder, exist_ok=True)
    
    trainer.evaluate(
        test_loader=validLoader, 
        checkpoint_path=swa_save_path,
        save_visuals=True,           
        output_dir=visual_folder    
    )
    export_evaluate(trainer, split_name="valid_swa_fixed")

if __name__ == "__main__":
    main()
