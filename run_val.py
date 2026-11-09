from ultralytics import YOLO
import os
import csv
import json

# ========== 配置 ==========
MODEL_PATH = 'runs/pose/fish3/weights/best.pt'
DATA_PATH = 'datasets/animalpose/animalpose.yaml'
IMG_SIZE = 640
DEVICE = 1
SAVE_DIR = 'runs/pose/fish3_val_results'

# ========== 创建保存目录 ==========
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== 加载模型 ==========
print(f"📦 加载模型：{MODEL_PATH}")
model = YOLO(MODEL_PATH)

# ========== 运行验证 ==========
print(f"🔍 开始验证...")
metrics = model.val(
    data=DATA_PATH,
    imgsz=IMG_SIZE,
    device=DEVICE,
    plots=True,
    save_json=True,
    verbose=True
)

# ========== 保存结果为 JSON ==========
json_path = os.path.join(SAVE_DIR, 'results.json')
results_dict = {
    'model': MODEL_PATH,
    'data': DATA_PATH,
    'imgsz': IMG_SIZE,
    'box': {
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'mAP50': float(metrics.box.map50),
        'mAP50-95': float(metrics.box.map)
    },
    'pose': {  # ✅ 改为 pose（不是 kpt）
        'precision': float(metrics.pose.mp),
        'recall': float(metrics.pose.mr),
        'mAP50': float(metrics.pose.map50),
        'mAP50-95': float(metrics.pose.map)
    }
}

with open(json_path, 'w') as f:
    json.dump(results_dict, f, indent=2)
print(f"✅ JSON 已保存：{json_path}")

# ========== 保存结果为 CSV ==========
csv_path = os.path.join(SAVE_DIR, 'results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Category', 'Metric', 'Value'])
    writer.writerow(['Box', 'Precision', f'{metrics.box.mp:.4f}'])
    writer.writerow(['Box', 'Recall', f'{metrics.box.mr:.4f}'])
    writer.writerow(['Box', 'mAP50', f'{metrics.box.map50:.4f}'])
    writer.writerow(['Box', 'mAP50-95', f'{metrics.box.map:.4f}'])
    writer.writerow(['Pose', 'Precision', f'{metrics.pose.mp:.4f}'])  # ✅ 改为 pose
    writer.writerow(['Pose', 'Recall', f'{metrics.pose.mr:.4f}'])    # ✅ 改为 pose
    writer.writerow(['Pose', 'mAP50', f'{metrics.pose.map50:.4f}'])  # ✅ 改为 pose
    writer.writerow(['Pose', 'mAP50-95', f'{metrics.pose.map:.4f}']) # ✅ 改为 pose
print(f"✅ CSV 已保存：{csv_path}")

# ========== 打印关键指标 ==========
print("\n" + "="*50)
print("📊 验证结果摘要")
print("="*50)
print(f"Box mAP50-95:      {metrics.box.map:.4f}")
print(f"Pose mAP50:        {metrics.pose.map50:.4f}")
print(f"Pose mAP50-95:     {metrics.pose.map:.4f}  ⭐ 核心指标")
print(f"Pose Precision:    {metrics.pose.mp:.4f}")
print(f"Pose Recall:       {metrics.pose.mr:.4f}")
print("="*50)
print(f"📁 结果保存至：{SAVE_DIR}")