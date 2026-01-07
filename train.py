import argparse
from dataset import*

def get_args():
    # Tham số bắt buộc nhập
    parser = argparse.ArgumentParser(description="Train, Pretrain hoặc Evaluate một model AI")
    parser.add_argument("--epoch", type=int, help="Số epoch để train")
    # parser.add_argument("--model", type=str, required=True, help="Đường dẫn đến model")
    parser.add_argument("--mode", type=str, choices=["train", "pretrain", "evaluate"], required=True, help="Chế độ: train hoặc pretrain hoặc evaluate")
    parser.add_argument("--data", type=str, required=True, help="Đường dẫn đến dataset đã giải nén")
    # Tham số trường hợp
    parser.add_argument("--checkpoint", type=str, help="Đường dẫn đến file checkpoint (chỉ dùng cho chế độ pretrain)")
    parser.add_argument("--augment", action='store_true', help="Bật Augmentation cho dữ liệu đầu vào")
    # Tham số mặc định(default)
    parser.add_argument("--saveas", type=str, help="Thư mục lưu checkpoint")
    parser.add_argument("--lr0", type=float, help="learning rate, default = 0.0001")
    parser.add_argument("--batchsize", type=int, help="Batch size, default = 8")

    parser.add_argument("--weight_decay", type=float,  help="weight_decay, default = 1e-6")
    parser.add_argument("--img_size", type=int, nargs=2,  help="Height and width of the image, default = [256, 256]")
    parser.add_argument("--numclass", type=int, help="shape of class, default = 1")
    parser.add_argument("--warmup", type=int, default=10, help="Số epoch để warm-up (augment nhẹ)")
    """
    # Với img_size, cách chạy: python script.py --img_size 256 256
    Nếu muốn nhập list dài hơn 3 phần tử, gõ 
    parser.add_argument("--img_size", type=int, nargs='+', default=[256, 256], help="Image dimensions")
    Chạy:
    python script.py --img_size 128 128 3
    """
    parser.add_argument("--loss", type=str, choices=["Dice_loss", "Hybric_loss", "BCEDice_loss", "BCEwDice_loss", "BCEw_loss", "SoftDice_loss", "Combo_loss", "Tversky_loss", "FocalTversky_loss" ], default="Combo_loss", help="Hàm loss sử dụng, default = Combo_loss")
    parser.add_argument("--optimizer", type=str, choices=["Adam", "SGD", "AdamW"], default="AdamW", help="Optimizer sử dụng, default = AdamW")
    args = parser.parse_args()
    
    # Kiểm tra logic tham số
    if args.mode in ["pretrain", "evaluate"] and not args.checkpoint:
        parser.error(f"--checkpoint là bắt buộc khi mode là '{args.mode}'")
        
    return args
def set_seed():
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# --- [THÊM HÀM NÀY] HÀM HỖ TRỢ ĐÓNG/MỞ BĂNG ---
def set_grad_status(model, freeze=True):
    """
    Hàm đóng băng hoặc mở băng Backbone/Encoder.
    Hỗ trợ cả Model Custom (self.backbone) và Model SMP (self.encoder).
    """
    target_module = None
    
    # 1. Kiểm tra nếu là Model Custom (PyramidCbamGateResNetUNet)
    if hasattr(model, 'backbone'):
        target_module = model.backbone
        name = "Backbone (ResNet)"
    # 2. Kiểm tra nếu là Model SMP (DeepLabV3+, Unet++, ...)
    elif hasattr(model, 'encoder'):
        target_module = model.encoder
        name = "Encoder (SMP)"
    
    if target_module:
        for param in target_module.parameters():
            param.requires_grad = not freeze # Freeze = True -> requires_grad = False
        
        status = "FROZEN ❄️" if freeze else "UNFROZEN 🔥"
        print(f"[INFO] {name} is now {status}")
    else:
        print("[WARNING] Could not find 'backbone' or 'encoder' to freeze!")
def model_factory(in_channels=3, num_classes=1):
    return smp.UnetPlusPlus(
        encoder_name="tu-resnest50d", 
        encoder_weights=None, # QUAN TRỌNG: Để None cho load nhanh, vì đằng nào cũng load checkpoint đè lên
        in_channels=in_channels,
        classes=num_classes,
        drop_path_rate=0.5
    )
def initialize_training_setup(args):
    from utils import get_loss_instance, _focal_tversky_global
    """
    Hàm khởi tạo dùng chung cho cả Train và Pretrain để tránh lặp code.
    Trả về: trainer, model, optimizer, criterion, scheduler
    """
    print(f"[INIT] Initializing Model, Optimizer, and Trainer...")
    
    # 1. Khởi tạo Model
    model = model_factory(in_channels=3, num_classes=1)
    # 2. Load Pretrained Weights (DDSM) nếu có
    # (Logic này dùng chung cho cả 2 mode đều tốt)
    ddsm_checkpoint_path = "best_model_cbis_ddsm.pth"
    if os.path.exists(ddsm_checkpoint_path):
        print(f"[TRANSFER] Loading weights from CBIS-DDSM: {ddsm_checkpoint_path}")
        try:
            state_dict = torch.load(ddsm_checkpoint_path)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            model.load_state_dict(state_dict)
            print("[TRANSFER] Weights loaded successfully! 🚀")
        except Exception as e:
            print(f"[ERROR] Weight mismatch: {e}")
    else:
        print(f"[INFO] No DDSM checkpoint found. Training from ImageNet/Scratch.")

    # 3. Optimizer & Scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    # 4. Loss Function
    criterion = get_loss_instance(args.loss)
    # Cập nhật tham số nếu là FocalTversky
    if args.loss == "FocalTversky_loss":
        alpha, beta, gamma = 0.4, 0.6, 1.33
        _focal_tversky_global.update_params(alpha=alpha, beta=beta, gamma=gamma)
        print(f"[CONFIG] Loss params updated: Alpha={alpha}, Beta={beta}, Gamma={gamma}")

    # 5. Trainer
    return model, opt, criterion, scheduler


def main(args):  
    print(f"\n[DEBUG TRAIN] args.loss bạn nhập từ bàn phím = {args.loss}")
    print("-" * 50)
    import numpy as np    
    from trainer import Trainer
    from model import Unet, unet_pyramid_cbam_gate, Swin_unet
    # from model import Swin_unet
    import optimizer as optimizer_module
    from dataset import get_dataloaders
    from result import export, export_evaluate
    global trainer
    from utils import _focal_tversky_global
    from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
    import shutil
    # from utils import loss_func
    from torch.optim.lr_scheduler import _LRScheduler
    print("-" * 50)
    print(f"[INFO] Mode: {args.mode.upper()}")
    print("-" * 50)
    import glob
    import os
    from dataset import EnsembleModel
    set_seed()
    if args.mode == "train":
        if not os.path.exists(BASE_OUTPUT):
            os.makedirs(BASE_OUTPUT)
        # Biến lưu kết quả 4 fold
        fold_scores = []
        # --- VÒNG LẶP 4 FOLD ---

        NUM_FOLDS = 4
        for fold_idx in range(NUM_FOLDS):
            print("\n" + "#"*60)
            print(f"### STARTING FOLD {fold_idx + 1}/{NUM_FOLDS} ###")
            print("#"*60 + "\n")
            model, opt, criterion, scheduler = initialize_training_setup(args)
            # trainer = Trainer(model=model, optimizer=opt, criterion=criterion_init, scheduler=scheduler_initial, patience=10, device=DEVICE)
            trainer = Trainer(
                model=model, 
                optimizer=opt, 
                criterion=criterion, 
                scheduler=scheduler, 
                patience=10, 
                device=DEVICE
            )
            # Reset best_val_loss cho fold mới
            trainer.best_val_loss = float('inf')
            # resume_checkpoint = None
            # --- GIAI ĐOẠN 1: FREEZE ENCODER ---
            # --- BƯỚC 1: ĐÓNG BĂNG ENCODER (Để Decoder làm quen ảnh mới trước) ---
            lr_stage1 = 1e-4
            print(f"\n[FOLD {fold_idx}] STEP 1: Freeze Encoder Training")
            print("\n" + "="*40)
            print(" TRANSFER LEARNING: INbreast Dataset")
            print(f" Strategy: Low LR {lr_stage1} + Frozen Encoder First|Train Decoder only")
            print("="*40)
            # [QUAN TRỌNG] Augmentation cho INbreast phải MẠNH vì dữ liệu ít
            trainLoader, validLoader, _ = get_dataloaders(aug_mode='strong', state='train', fold_idx=fold_idx)
            set_grad_status(model, freeze=True) # Hàm có sẵn của bạn
            trainer.optimizer.param_groups[0]['lr'] = lr_stage1 # Decoder học nhanh hơn chút
            trainer.num_epochs = 150 # Chạy tầm 20 epoch
            trainer.patience = 30
            trainer.scheduler = None # Không cần giảm LR đoạn này
            # Train nhẹ
            trainer.train(trainLoader, validLoader, resume_path=None)
            # =========================================================
            # # --- BƯỚC 2: MỞ BĂNG TOÀN BỘ (FULL FINE-TUNE) ---
            # =========================================================     
            print(f"\n[FOLD {fold_idx}] STEP 2: Full Fine-tuning, Unfreezing All Layers... Fine-tuning with Low LR.")             
            set_grad_status(model, freeze=False) # Mở khóa
            # Update Loss Params (Nên làm mới mỗi fold để chắc chắn)
            if args.loss == "FocalTversky_loss":
                _focal_tversky_global.update_params(alpha=0.4, beta=0.6, gamma=1)
                trainer.best_val_loss = float('inf')
            step1_ckpt = "best_dice_mass_model.pth"
            if os.path.exists(step1_ckpt):
                print(f"[FOLD {fold_idx}] Loading best model from Step 1 manually...")
                trainer.load_checkpoint(step1_ckpt)
            else:
                print(f"[WARNING] No checkpoint found at {step1_ckpt}. Training from scratch/ImageNet.")
            # Reset LR về mức siêu thấp
            lr_stage2 = 1e-5 
            weight_decay_stage2 = 1e-2
            trainer.optimizer.param_groups[0]['lr'] = lr_stage2
            trainer.optimizer.param_groups[0]['weight_decay'] = weight_decay_stage2
            print(f"[CONFIG] Updated Optimizer: LR = {lr_stage2} | Weight Decay = {weight_decay_stage2}")
            # Gán lại Scheduler để giảm LR nếu kẹt
            trainer.scheduler = scheduler
            
            trainer.num_epochs = NUM_EPOCHS # Chạy lâu
            trainer.patience = 25    # Kiên nhẫn
            trainer.early_stop_counter = 0 # Reset đếm
            trainer.train(trainLoader, validLoader, resume_path=None)
            # --- LƯU KẾT QUẢ FOLD ---
            best_dice = trainer.best_dice_mass
            print(f"--> [RESULT] Fold {fold_idx} Best Dice: {best_dice:.4f}")
            fold_scores.append(best_dice)
            # --- XUẤT KẾT QUẢ (EXPORT) ---
            print(f"\n[INFO] Exporting Fine-tuned Results for FOLD {fold_idx}...")
            # [QUAN TRỌNG] Gọi hàm export MỚI với tham số fold_idx
            # Hàm này sẽ tự động tạo folder 'output/fold_X' và move file model + csv vào đó
            export(trainer, fold_idx=fold_idx)
            # =========================================================
            # GIAI ĐOẠN 4: SWA (STOCHASTIC WEIGHT AVERAGING)
            # =========================================================
            # Chỉ chạy SWA nếu đang dùng FocalTversky (chiến lược của bạn)
            if args.loss == "FocalTversky_loss":
                print("\n" + "="*40)
                print(" GIAI ĐOẠN 4: SWA FINETUNING (The Secret Weapon)")
                print(" Strategy: Constant LR | No Early Stop | 5 Epochs")
                print("="*40)
                # 1. Định nghĩa thư mục Fold hiện tại
                fold_dir = os.path.join(BASE_OUTPUT, f"fold_{fold_idx}")
                os.makedirs(fold_dir, exist_ok=True)
                # current_fold = fold if 'fold' in locals() else None 
                # 1. QUAN TRỌNG: Load lại BEST MODEL của GD3 (Không dùng model cuối cùng)
                # best_model_path = "best_dice_mass_model.pth"
                path_to_best_model = os.path.join(fold_dir, "best_dice_mass_model.pth")
                if not os.path.exists(path_to_best_model):
                    best_ep = trainer.best_epoch_dice
                    best_d = trainer.best_dice_mass
                    folder_name = f"output_epoch{best_ep}_diceMass{best_d:.4f}"
                    path_to_best_model = os.path.join(BASE_OUTPUT, folder_name, "best_dice_mass_model.pth")

                if os.path.exists(path_to_best_model):
                    print(f"[INFO] Loading BEST model from previous stage for SWA: {path_to_best_model}")
                    trainer.load_checkpoint(path_to_best_model)
                else:
                    print(f"[WARNING] Could not find {path_to_best_model}. Using current weights.")

                # 2. Khởi tạo SWA
                swa_model = AveragedModel(trainer.model)
                # LR cho SWA: Cao hơn GD3 một chút để thoát hố (5e-5 là an toàn với AdamW)
                swa_lr = 5e-5 
                swa_scheduler = SWALR(trainer.optimizer, swa_lr=swa_lr, anneal_epochs=3)
                print(f"[CONFIG] SWA Scheduler set. LR: {swa_lr}")

                # 3. Cấu hình vòng lặp SWA
                SWA_EPOCHS = 5 # Chạy cố định
                trainer.patience = 999 # Tắt Early Stop
                trainer.early_stop_counter = 0
                
                # Chúng ta sẽ dùng lại hàm train() của Trainer nhưng chạy từng epoch một
                # để chèn logic update_parameters vào giữa.
                
                print("[INFO] Starting SWA Loop...")
                for epoch in range(SWA_EPOCHS):
                    # Hack: Set epoch = 1 để Trainer chạy 1 vòng rồi thoát ra
                    trainer.num_epochs = 1 
                    trainer.start_epoch = 0 
                    # Gán scheduler SWA vào trainer
                    trainer.scheduler = swa_scheduler
                    
                    # Train 1 epoch (Không load checkpoint, chạy tiếp từ bộ nhớ)
                    # Lưu ý: Trainer sẽ in ra log validation, cứ kệ nó.
                    print(f"\n[SWA] Epoch {epoch+1}/{SWA_EPOCHS}")
                    trainer.train(trainLoader, validLoader, resume_path=None) # Dùng trainLoader (đã strong aug)
                    
                    # Cập nhật trọng số trung bình
                    swa_model.update_parameters(trainer.model)
                    
                    # Step Scheduler
                    swa_scheduler.step()
                    
                # 4. Cập nhật Batch Norm (Bước bắt buộc)
                print("\n[INFO] Updating Batch Normalization statistics for SWA Model...")
                update_bn(trainLoader, swa_model, device=DEVICE)

                # 5. Lưu và Đánh giá SWA Model
                swa_save_path = os.path.join(fold_dir, "best_model_swa.pth")
                print(f"[INFO] Saving SWA Model to {swa_save_path}")
                swa_checkpoint = {
                    'epoch': SWA_EPOCHS,
                    'model_state_dict': swa_model.state_dict(),         # <--- Đã sửa để khớp tên layer
                    'optimizer_state_dict': trainer.optimizer.state_dict(), # Để không lỗi optimizer
                    
                    # Các chỉ số thống kê (Lấy từ trainer hiện tại để lưu làm kỷ niệm)
                    'best_dice_mass': trainer.best_dice_mass,
                }
                torch.save(swa_checkpoint, swa_save_path)
                # export(trainer)
                # Đánh giá Model SWA
                print("\n[INFO] Evaluating SWA Model...")
                # Gán model SWA vào trainer để evaluate
                trainer.model = swa_model
                
                visual_folder = os.path.join(fold_dir, "prediction_images_swa")
                os.makedirs(visual_folder, exist_ok=True)
                
                trainer.evaluate(
                    test_loader=validLoader, 
                    checkpoint_path=swa_save_path,
                    save_visuals=True,          
                    output_dir=visual_folder    
                )
                export_evaluate(trainer, split_name="valid_swa", fold_idx=fold_idx)
        # --- HẾT VÒNG FOR (KẾT THÚC 5 FOLD) ---
        # Tính toán trung bình kết quả tại đây
        if fold_scores:
            mean_score = np.mean(fold_scores)
            std_score = np.std(fold_scores)
            print("\n" + "="*60)
            print(f" FINAL 5-FOLD CV RESULTS")
            print(f" Scores per fold: {fold_scores}")
            print(f" Average Dice: {mean_score:.4f} (+/- {std_score:.4f})")
            print("="*60)    
    # (Giữ nguyên phần pretrain/evaluate)
    elif args.mode == "pretrain":
        print(f"\n[INFO] Mode: PRETRAIN (Single Run)")
        model, opt, criterion, scheduler = initialize_training_setup(args)
        trainer = Trainer(
                model=model, 
                optimizer=opt, 
                criterion=criterion, 
                scheduler=scheduler, 
                patience=20, 
                device=DEVICE
            )
        aug_type = 'strong' if args.augment else 'none'
        trainLoader, validLoader, _ = get_dataloaders(aug_mode=aug_type, state='train', fold_idx=None) 
        trainer.train(trainLoader, validLoader, resume_path=args.checkpoint)
        export(trainer, fold_idx=None) 
    elif args.mode == "evaluate":
        print(f"[INFO] Mode: EVALUATING FULL DATASET")
        
        _, _, testLoader = get_dataloaders(aug_mode='none', state='evaluate')
        model_paths = []
        base_checkpoint_path = args.checkpoint 
        NUM_FOLDS_TO_EVAL = 4
        print(f"[SEARCH] Looking for models in: {base_checkpoint_path}")
        
        for i in range(NUM_FOLDS_TO_EVAL):
            # Tìm file .pth trong mỗi fold (dùng * để bỏ qua phần tên epoch dài dòng)
            search_pattern = os.path.join(base_checkpoint_path, f"fold_{i}", "**", "best_dice_mass_model.pth")
            files = glob.glob(search_pattern, recursive=True)
            
            if files:
                model_paths.append(files[0])
                print(f"Fold {i}: Found {files[0]}")
            else:
                print(f"  ! Fold {i}: Warning - File not found at {search_pattern}")
        if len(model_paths) == 0:
            raise ValueError("Không tìm thấy model nào! Kiểm tra lại đường dẫn output.")
        # 4. KHỞI TẠO ENSEMBLE MODEL
        print(f"[ENSEMBLE] Initializing Ensemble with {len(model_paths)} models...")
        ensemble_model = EnsembleModel(
            model_class=model_factory,  # Truyền hàm factory
            checkpoint_paths=model_paths, 
            device=DEVICE,
            in_channels=3,
            num_classes=1
        )
        
        trainer = Trainer(model=ensemble_model, device=DEVICE)
        ensemble_output_dir = os.path.join(base_checkpoint_path, "ensemble_predictions_final")
        os.makedirs(ensemble_output_dir, exist_ok=True)
        print(f"[EXEC] Running Inference & Visualization...")
        # Gọi hàm evaluate có sẵn của Trainer
        trainer.evaluate(
            test_loader=testLoader, 
            checkpoint_path=None, # Không cần load path vì Ensemble đã load rồi
            save_visuals=True, 
            output_dir=ensemble_output_dir
        )
        export_evaluate(trainer, split_name="final_ensemble_test", fold_idx="ensemble")
        print("[DONE] Ensemble Evaluation Finished.")

if __name__ == "__main__":
    args = get_args()
    main(args)
