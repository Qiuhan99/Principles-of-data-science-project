"""
简化的测试集预测脚本
可以直接导入训练脚本中的模型类，避免重复定义

使用方法:
    python predict_test_set_simple.py --test_dir /path/to/test --model_path model.pth
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 尝试从训练脚本导入模型类
try:
    # 尝试导入Colab版本
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dual_path_analysis_colab import DualPathModel, SwinBackbone, NUM_CLASSES, CLASSES as TRAIN_CLASSES
    print("✓ 从 dual_path_analysis_colab 导入模型类")
except ImportError:
    try:
        # 尝试导入标准版本
        from dual_path_analysis import DualPathModel, SwinBackbone, NUM_CLASSES, CLASSES as TRAIN_CLASSES
        print("✓ 从 dual_path_analysis 导入模型类")
    except ImportError:
        print("警告: 无法导入训练脚本，将使用内置模型定义")
        # 如果导入失败，使用内置定义（简化版）
        from transformers import SwinModel, SwinConfig
        import torch.nn as nn
        
        class SwinBackbone(nn.Module):
            def __init__(self, model_name='microsoft/swin-tiny-patch4-window7-224'):
                super().__init__()
                config = SwinConfig.from_pretrained(model_name)
                self.swin = SwinModel.from_pretrained(model_name, config=config)
            
            def forward(self, x):
                outputs = self.swin(x)
                if hasattr(outputs, 'last_hidden_state'):
                    return outputs.last_hidden_state.mean(dim=1)
                return outputs.pooler_output
        
        class DualPathModel(nn.Module):
            def __init__(self, num_classes=7):
                super().__init__()
                self.global_backbone = SwinBackbone()
                self.local_backbone = SwinBackbone()
                self.global_backbone.eval()
                with torch.no_grad():
                    dummy_input = torch.randn(1, 3, 224, 224)
                    feature_dim = self.global_backbone(dummy_input).shape[1]
                self.global_classifier = nn.Linear(feature_dim, num_classes)
                self.local_classifier = nn.Linear(feature_dim, num_classes)
            
            def forward(self, full_image, local_image=None):
                global_features = self.global_backbone(full_image)
                global_logits = self.global_classifier(global_features)
                if local_image is None:
                    local_image = full_image
                local_features = self.local_backbone(local_image)
                local_logits = self.local_classifier(local_features)
                return global_logits, local_logits, global_features, local_features
        
        TRAIN_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
        NUM_CLASSES = len(TRAIN_CLASSES)

# 提交时的类别（8个，包括NoF）
SUBMIT_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

# Colab环境检测
try:
    import google.colab
    IN_COLAB = True
    BASE_DIR = '/content'
except ImportError:
    IN_COLAB = False
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class TestDataset(Dataset):
    """测试集数据集类"""
    
    def __init__(self, test_dir: str, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = []
        
        # 支持的图片格式
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG']
        
        # 搜索所有图片文件
        if os.path.isdir(test_dir):
            for ext in image_extensions:
                pattern1 = os.path.join(test_dir, f'*{ext}')
                pattern2 = os.path.join(test_dir, '**', f'*{ext}')
                self.image_files.extend(glob.glob(pattern1))
                self.image_files.extend(glob.glob(pattern2, recursive=True))
        elif os.path.isfile(test_dir):
            self.image_files = [test_dir]
        
        # 去重并排序
        self.image_files = sorted(list(set(self.image_files)))
        
        if not self.image_files:
            print(f"⚠️  警告: 在 {test_dir} 中未找到任何图片文件")
        else:
            print(f"✓ 找到 {len(self.image_files)} 张测试图片")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img_filename = os.path.basename(img_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return {'image': image, 'filename': img_filename, 'full_path': img_path}
        except Exception as e:
            print(f"⚠️  警告: 无法加载图片 {img_path}: {e}")
            placeholder = Image.new('RGB', (224, 224), color='gray')
            if self.transform:
                placeholder = self.transform(placeholder)
            return {'image': placeholder, 'filename': img_filename, 'full_path': img_path}


def load_model(model_path: str, device='cuda'):
    """加载训练好的模型"""
    print(f"\n📦 加载模型: {model_path}")
    
    # 尝试多个可能的路径
    possible_paths = [
        model_path,
        os.path.join(BASE_DIR, model_path),
        os.path.join('/content', model_path),
        os.path.join('/content/drive/MyDrive', model_path),
        os.path.join('//content/drive/MyDrive/fish dataset/test_stg1', model_path),
    ]
    
    actual_path = None
    for path in possible_paths:
        if os.path.exists(path):
            actual_path = path
            break
    
    if actual_path is None:
        print("❌ 错误: 模型文件不存在")
        print("尝试过的路径:")
        for path in possible_paths:
            print(f"  - {path}")
        return None
    
    print(f"✓ 找到模型文件: {actual_path}")
    
    # 创建模型
    model = DualPathModel(num_classes=NUM_CLASSES)
    
    # 加载权重
    try:
        state_dict = torch.load(actual_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✓ 模型权重加载成功")
    except Exception as e:
        print(f"❌ 错误: 加载模型权重失败: {e}")
        return None
    
    model.to(device)
    model.eval()
    return model


def predict_test_set(model, test_loader, device='cuda', use_ensemble=True):
    """对测试集进行预测"""
    model.eval()
    predictions = {}
    
    print(f"\n🔮 开始预测 (使用{'集成' if use_ensemble else '全局'}预测)...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch['image'].to(device)
            filenames = batch['filename']
            
            # 前向传播（测试集没有BBox，使用整张图片作为全局和局部）
            global_logits, local_logits, _, _ = model(images, images)
            
            # 转换为概率
            global_probs = F.softmax(global_logits, dim=1)
            local_probs = F.softmax(local_logits, dim=1)
            
            # 集成预测
            if use_ensemble:
                ensemble_probs = (global_probs + local_probs) / 2.0
            else:
                ensemble_probs = global_probs
            
            # 存储预测结果
            probs_np = ensemble_probs.cpu().numpy()
            for i, filename in enumerate(filenames):
                predictions[filename] = probs_np[i]
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  进度: {batch_idx + 1}/{len(test_loader)} 批次")
    
    print(f"✓ 预测完成，共 {len(predictions)} 张图片")
    return predictions


def convert_to_submit_format(predictions: dict) -> pd.DataFrame:
    """将预测结果转换为提交格式"""
    rows = []
    
    for filename, probs in predictions.items():
        # 创建8个类别的概率数组
        submit_probs = np.zeros(8)
        
        # 映射7个训练类别到8个提交类别
        for i, train_class in enumerate(TRAIN_CLASSES):
            if train_class in SUBMIT_CLASSES:
                submit_idx = SUBMIT_CLASSES.index(train_class)
                submit_probs[submit_idx] = probs[i]
        
        # NoF概率：1 - max(其他类别概率)
        max_other_prob = np.max(submit_probs[:4]) + np.max(submit_probs[5:])
        nof_prob = max(0.0, min(1.0, 1.0 - max_other_prob))
        submit_probs[4] = nof_prob
        
        # 归一化
        submit_probs = submit_probs / (submit_probs.sum() + 1e-10)
        
        # 概率裁剪
        submit_probs = np.clip(submit_probs, 1e-15, 1 - 1e-15)
        
        rows.append({
            '图像': filename,
            'ALB': submit_probs[0],
            'BET': submit_probs[1],
            'DOL': submit_probs[2],
            'LAG': submit_probs[3],
            'NoF': submit_probs[4],
            '其他': submit_probs[5],
            'SHARK': submit_probs[6],
            'YFT': submit_probs[7]
        })
    
    df = pd.DataFrame(rows)
    column_order = ['图像', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', '其他', 'SHARK', 'YFT']
    return df[column_order]


def main():
    parser = argparse.ArgumentParser(description='测试集预测脚本（简化版）')
    parser.add_argument('--test_dir', type=str, required=True,
                        help='测试图片目录路径')
    parser.add_argument('--model_path', type=str, default='dual_path_model.pth',
                        help='训练好的模型路径')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='输出CSV文件路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--no_ensemble', action='store_true',
                        help='不使用集成预测')
    
    args = parser.parse_args()
    
    # 检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  使用设备: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载模型
    model = load_model(args.model_path, device)
    if model is None:
        return
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建测试数据集
    print(f"\n📂 加载测试集: {args.test_dir}")
    test_dataset = TestDataset(args.test_dir, transform=transform)
    
    if len(test_dataset) == 0:
        print("❌ 错误: 未找到任何测试图片")
        return
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if IN_COLAB else 2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 进行预测
    predictions = predict_test_set(
        model, test_loader, device, 
        use_ensemble=not args.no_ensemble
    )
    
    # 转换为提交格式
    print("\n📝 生成提交文件...")
    submission_df = convert_to_submit_format(predictions)
    
    # 保存CSV文件
    output_path = args.output
    submission_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✓ 提交文件已保存到: {output_path}")
    print(f"  共 {len(submission_df)} 张图片")
    
    # 显示预览
    print("\n📊 前5行预览:")
    print(submission_df.head().to_string())
    
    # 验证概率和
    prob_sums = submission_df.iloc[:, 1:].sum(axis=1)
    print(f"\n✅ 概率和检查:")
    print(f"  最小: {prob_sums.min():.6f}")
    print(f"  最大: {prob_sums.max():.6f}")
    print(f"  平均: {prob_sums.mean():.6f}")
    
    if IN_COLAB:
        print("\n💡 Colab提示:")
        print("  下载文件: from google.colab import files; files.download('submission.csv')")


if __name__ == '__main__':
    main()

