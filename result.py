
from config import*
from train import*
def tensor_to_float(value):
        if isinstance(value, torch.Tensor):
            return value.cpu().item()  # Chuyển tensor về CPU và lấy giá trị float
        elif isinstance(value, list):
            return [tensor_to_float(v) for v in value]  # Xử lý danh sách các tensor
        return value  # Nếu không phải tensor, giữ nguyên
def export(trainer, fold_idx=None):
    source_file1='last_model.pth'
    source_file2 = 'best_dice_mass_model.pth' # Tên file best model mới
    source_file3 = 'best_iou_mass_model.pth'  # Tên file best iou mới (nếu muốn move cả cái này)
    # Lấy thông tin từ trainer
    best_ep = trainer.best_epoch_dice
    best_d = trainer.best_dice_mass
    # --- SỬA LOGIC TẠO FOLDER ---
    if fold_idx is not None:
        # Nếu chạy K-Fold, lưu vào folder định danh: fold_0, fold_1...
        # Bên trong vẫn giữ tên file gốc để dễ quản lý
        output_folder_fold = os.path.join(BASE_OUTPUT, f"fold_{fold_idx}")
        path=f"output_epoch{best_ep}_diceMass{best_d:.4f}"
        output_folder = os.path.join(output_folder_fold, path)
    else:
        path=f"output_epoch{best_ep}_diceMass{best_d:.4f}"
        output_folder = os.path.join(BASE_OUTPUT,path)
    os.makedirs(output_folder, exist_ok=True)
    print(f"[EXPORT] Saving results to: {output_folder}")
    # --- PHẦN DI CHUYỂN FILE (Giữ nguyên logic nhưng thêm try-except cho an toàn) ---
    def safe_move(filename):
        if os.path.exists(filename):
            dst = os.path.join(output_folder, filename)
            if os.path.exists(dst):
                os.remove(dst) # Xóa file cũ ở đích nếu có để ghi đè
            try:
                shutil.move(filename, output_folder)
                print(f"  -> Moved {filename}")
            except Exception as e:
                print(f"  [Error] Moving {filename}: {e}")
    # Di chuyển
    safe_move(source_file1)
    safe_move(source_file2)
    safe_move(source_file3)

    
    # --- XỬ LÝ SỐ LIỆU (HISTORY) ---
    checkpoint_path = os.path.join(output_folder, source_file1) # Load từ last model
    if not os.path.exists(checkpoint_path):
        print(f"[WARN] Không tìm thấy checkpoint để xuất lịch sử tại {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Lấy history dictionary
    history = checkpoint.get('history', {})
    
    # Trích xuất dữ liệu (dùng key mới trong trainer.py)
    train_losses = tensor_to_float(history.get('train_loss', []))
    val_losses = tensor_to_float(history.get('val_loss', []))
    
    train_dice_mass = tensor_to_float(history.get('train_dice_mass', []))
    val_dice_mass = tensor_to_float(history.get('val_dice_mass', []))
    train_dice_norm = tensor_to_float(history.get('train_dice_norm', []))
    val_dice_norm = tensor_to_float(history.get('val_dice_norm', []))
    
    train_iou_mass = tensor_to_float(history.get('train_iou_mass', []))
    val_iou_mass = tensor_to_float(history.get('val_iou_mass', []))
    train_iou_norm = tensor_to_float(history.get('train_iou_norm', []))
    val_iou_norm = tensor_to_float(history.get('val_iou_norm', []))

    # Lấy thông tin Best
    best_dice_mass = tensor_to_float(checkpoint.get('best_dice_mass', 0))
    best_iou_mass = tensor_to_float(checkpoint.get('best_iou_mass', 0))
    best_epoch_dice = tensor_to_float(checkpoint.get('best_epoch_dice', 0))
    best_epoch_iou = tensor_to_float(checkpoint.get('best_epoch_iou', 0))
    
    epoch = checkpoint.get('epoch', 0)
    epochs = list(range(1, epoch + 1))
    
    # Tạo DataFrame đầy đủ
    data = {
        'epoch': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        # Mass Metrics
        'train_dice_mass': train_dice_mass,
        'val_dice_mass': val_dice_mass,
        'train_iou_mass': train_iou_mass,
        'val_iou_mass': val_iou_mass,
        # Normal Metrics
        'train_dice_norm': train_dice_norm,
        'val_dice_norm': val_dice_norm,
        'train_iou_norm': train_iou_norm,
        'val_iou_norm': val_iou_norm,
        # Best info (lặp lại cho đủ hàng)
        'best_dice_mass': [best_dice_mass] * len(epochs),
        'best_iou_mass': [best_iou_mass] * len(epochs),
        'best_epoch_dice': [best_epoch_dice] * len(epochs),
        'best_epoch_iou': [best_epoch_iou] * len(epochs),
    }
    
    # Đảm bảo độ dài các mảng bằng nhau (tránh lỗi nếu checkpoint lưu thiếu bước cuối)
    min_len = min(len(v) for k, v in data.items() if isinstance(v, list))
    for k in data:
        if isinstance(data[k], list):
            data[k] = data[k][:min_len]

    new_data = pd.DataFrame(data)
    
    # Lưu CSV
    csv_path = os.path.join(output_folder, 'training_history.csv')
    new_data.to_csv(csv_path, index=False)
    print(f"[INFO] Training history saved to {csv_path}")
    
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')  # Hoặc 'latin1', 'windows-1252'
    # df.info()

    # Plot Losses
    plt.figure(figsize=(18, 6)) # Tăng chiều rộng để dễ nhìn
    
    max_epoch = df['epoch'].max()
    max_epoch = int(df['epoch'].max())
    step = int(max(1, max_epoch // 10))
    xticks_range = range(0, max_epoch + 1, step)

    # 1. Plot Losses
    plt.subplot(1, 3, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', color='blue')
    plt.plot(df['epoch'], df['val_loss'], label='Valid Loss', color='orange')
    plt.title('Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 2. Plot Dice (So sánh Mass và Norm)
    plt.subplot(1, 3, 2)
    # Mass - Nét liền
    plt.plot(df['epoch'], df['train_dice_mass'], label='Train Mass', color='green')
    plt.plot(df['epoch'], df['val_dice_mass'], label='Valid Mass', color='red')
    # Norm - Nét đứt (để xem model có bị ảo giác không)
    plt.plot(df['epoch'], df['train_dice_norm'], label='Train Norm', color='lightgreen', linestyle='--')
    plt.plot(df['epoch'], df['val_dice_norm'], label='Valid Norm', color='salmon', linestyle='--')
    
    plt.title(f'Dice Coefficients (Best Mass: {best_dice_mass:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.ylim(0, 1.05) # Dice max là 1
    plt.grid(True, linestyle='--', alpha=0.5)

    # 3. Plot IoU (So sánh Mass và Norm)
    plt.subplot(1, 3, 3)
    plt.plot(df['epoch'], df['train_iou_mass'], label='Train Mass', color='purple')
    plt.plot(df['epoch'], df['val_iou_mass'], label='Valid Mass', color='brown')
    plt.plot(df['epoch'], df['train_iou_norm'], label='Train Norm', color='violet', linestyle='--')
    plt.plot(df['epoch'], df['val_iou_norm'], label='Valid Norm', color='peru', linestyle='--')
    
    plt.title(f'IoU (Best Mass: {best_iou_mass:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    output_metric = os.path.join(output_folder, "metrics_chart.png")
    plt.savefig(output_metric, dpi=300)
    # plt.show() # Comment lại nếu chạy trên server không có màn hình
    plt.close()
def export_evaluate(trainer, split_name="test", fold_idx=None): # <--- Thêm fold_idx
    # --- SỬA LOGIC FOLDER ---
    if fold_idx == "ensemble":
        # Nếu là ensemble, lưu chung vào folder chứa ảnh predict cho gọn
        output_folder = os.path.join(BASE_OUTPUT, split_name)
    elif fold_idx is not None:
        output_folder = os.path.join(BASE_OUTPUT, f"fold_{fold_idx}")
    else:
        output_folder = BASE_OUTPUT
        
    os.makedirs(output_folder, exist_ok=True)
    
    # Lấy dữ liệu từ trainer
    df = pd.DataFrame({
        'ImagePath': trainer.path_list,
        'Type': trainer.type_list, 
        'Dice': trainer.dice_list,
        'IoU': trainer.iou_list
    })
    
    # Đặt tên file
    result_csv = f"{split_name}_metrics_details.csv" 
    
    output_result = os.path.join(output_folder, result_csv)
    df.to_csv(output_result, index=False)
    print(f"[INFO] Evaluation details saved to {output_result}")
        
# def export_evaluate(trainer):
#     output_folder = BASE_OUTPUT
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Lấy dữ liệu từ trainer (các list này được tạo trong hàm evaluate)
#     df = pd.DataFrame({
#         'ImagePath': trainer.path_list,
#         'Type': trainer.type_list,      # <--- Cột mới: Normal hoặc Mass
#         'Dice': trainer.dice_list,
#         'IoU': trainer.iou_list
#     })
    
#     # Thêm cột phân loại Mass/Normal để dễ lọc
#     # Giả sử ImagePath chứa tên file, ta không biết logic label ở đây
#     # Nhưng trainer.evaluate đã print report, file csv này lưu raw data từng ảnh
    
#     result_csv = "test_metrics_details.csv"
#     output_result = os.path.join(output_folder, result_csv)
#     df.to_csv(output_result, index=False)
#     print(f"[INFO] Evaluation details saved to {output_result}")
