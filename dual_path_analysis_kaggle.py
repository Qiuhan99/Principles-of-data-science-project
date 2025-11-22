"""
双路径鱼类分类模型分析脚本 - Kaggle版本
适配Kaggle环境的版本，自动处理路径和GPU

Kaggle使用说明:
1. 将数据集添加到Notebook的Input中
2. 数据集路径通常在 /kaggle/input/dataset-name/
3. 输出文件保存在 /kaggle/working/
"""

import json
import os
import glob
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import SwinModel, SwinConfig
import matplotlib.pyplot as plt

# 设置类别
CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT']
NUM_CLASSES = len(CLASSES)

# Kaggle环境检测
IN_KAGGLE = os.path.exists('/kaggle/input')
if IN_KAGGLE:
    BASE_DIR = '/kaggle/working'
    INPUT_DIR = '/kaggle/input'
    WORKING_DIR = '/kaggle/working'
    print("=" * 60)
    print("检测到Kaggle环境")
    print(f"输入目录: {INPUT_DIR}")
    print(f"工作目录: {WORKING_DIR}")
    print("=" * 60)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_DIR = BASE_DIR
    WORKING_DIR = BASE_DIR
    print("本地环境，使用当前目录")


class FishDataset(Dataset):
    """鱼类数据集类，支持全局和局部路径"""
    
    def __init__(self, data_dir: str, json_files: List[str], transform=None, expand_ratio=0.2, skip_missing=True):
        """
        Args:
            data_dir: 数据目录
            json_files: JSON标注文件列表
            transform: 图像变换
            expand_ratio: BBox扩展比例（默认20%）
            skip_missing: 是否跳过缺失的图片（默认True）
        """
        self.data_dir = data_dir
        self.transform = transform
        self.expand_ratio = expand_ratio
        self.skip_missing = skip_missing
        self.samples = []
        
        # 加载所有标注数据
        for json_file in json_files:
            if not os.path.exists(json_file):
                print(f"警告: 文件不存在 {json_file}")
                continue
                
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    filename = item['filename']
                    annotations = item['annotations']
                    
                    # 获取图片的类别（从第一个标注或文件名推断）
                    if annotations:
                        label = CLASSES.index(annotations[0]['class'])
                    else:
                        # 如果没标注，从文件名推断
                        for cls in CLASSES:
                            if cls in filename:
                                label = CLASSES.index(cls)
                                break
                        else:
                            continue
                    
                    # 存储每个标注框作为一个样本
                    for ann in annotations:
                        # 检查图片文件是否存在（预检查）
                        img_path = os.path.join(self.data_dir, filename)
                        if not os.path.exists(img_path):
                            # 尝试其他可能的路径（Kaggle适配）
                            img_name = os.path.basename(filename)
                            possible_paths = [
                                img_path,
                                os.path.join(self.data_dir, img_name),
                                os.path.join(BASE_DIR, img_name),
                                os.path.join(BASE_DIR, filename),
                                # Kaggle输入目录路径
                                os.path.join(INPUT_DIR, img_name),
                                os.path.join(INPUT_DIR, filename),
                            ]
                            
                            # 尝试在所有输入数据集中查找
                            if IN_KAGGLE:
                                for dataset_name in os.listdir(INPUT_DIR):
                                    dataset_path = os.path.join(INPUT_DIR, dataset_name)
                                    if os.path.isdir(dataset_path):
                                        possible_paths.extend([
                                            os.path.join(dataset_path, filename),
                                            os.path.join(dataset_path, img_name),
                                            os.path.join(dataset_path, 'images', filename),
                                            os.path.join(dataset_path, 'images', img_name),
                                        ])
                            
                            found = False
                            for path in possible_paths:
                                if os.path.exists(path):
                                    found = True
                                    break
                            
                            # 如果skip_missing为True且图片不存在，跳过这个样本
                            if self.skip_missing and not found:
                                continue
                        
                        self.samples.append({
                            'filename': filename,
                            'label': label,
                            'bbox': {
                                'x': ann['x'],
                                'y': ann['y'],
                                'width': ann['width'],
                                'height': ann['height']
                            }
                        })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = os.path.join(self.data_dir, sample['filename'])
        
        # 检查图片文件是否存在
        if not os.path.exists(img_path):
            # 尝试其他可能的路径
            img_name = os.path.basename(img_path)
            possible_paths = [
                img_path,  # 原始路径
                os.path.join(self.data_dir, img_name),
                os.path.join(BASE_DIR, img_name),
                os.path.join(BASE_DIR, sample['filename']),
                # Kaggle输入目录路径
                os.path.join(WORKING_DIR, img_name),
                os.path.join(WORKING_DIR, sample['filename']),
            ]
            
            # 尝试在所有输入数据集中查找
            if IN_KAGGLE:
                for dataset_name in os.listdir(INPUT_DIR):
                    dataset_path = os.path.join(INPUT_DIR, dataset_name)
                    if os.path.isdir(dataset_path):
                        possible_paths.extend([
                            os.path.join(dataset_path, sample['filename']),
                            os.path.join(dataset_path, img_name),
                            os.path.join(dataset_path, 'images', sample['filename']),
                            os.path.join(dataset_path, 'images', img_name),
                        ])
            
            # 如果文件名包含类别（如 ALB/img_xxx.jpg），尝试不同的基础路径
            if '/' in sample['filename']:
                parts = sample['filename'].split('/')
                if len(parts) >= 2:
                    category = parts[0]
                    img_file = parts[-1]
                    # 尝试不同的基础路径
                    base_paths = [
                        self.data_dir,
                        BASE_DIR,
                        INPUT_DIR,
                    ]
                    # 添加所有输入数据集路径
                    if IN_KAGGLE:
                        for dataset_name in os.listdir(INPUT_DIR):
                            dataset_path = os.path.join(INPUT_DIR, dataset_name)
                            if os.path.isdir(dataset_path):
                                base_paths.append(dataset_path)
                    
                    for base in base_paths:
                        possible_paths.append(os.path.join(base, category, img_file))
                        possible_paths.append(os.path.join(base, 'images', category, img_file))
                        possible_paths.append(os.path.join(base, img_file))
            
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    found = True
                    break
            
            if not found:
                # 如果图片不存在，创建占位图像
                print(f"警告: 图片文件不存在 {sample['filename']}，使用占位图像")
                full_image = Image.new('RGB', (800, 600), color='gray')
        
        # 加载完整图像（全局路径）
        try:
            full_image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 如果加载失败，跳过这个样本
            print(f"警告: 无法加载图片 {sample['filename']}: {e}")
            return None
        
        # 获取BBox并扩展20%
        bbox = sample['bbox']
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # 扩展BBox
        expand_w = w * self.expand_ratio
        expand_h = h * self.expand_ratio
        x_expanded = max(0, x - expand_w / 2)
        y_expanded = max(0, y - expand_h / 2)
        w_expanded = min(full_image.width - x_expanded, w + expand_w)
        h_expanded = min(full_image.height - y_expanded, h + expand_h)
        
        # 裁剪局部图像（局部路径）
        local_image = full_image.crop((
            int(x_expanded),
            int(y_expanded),
            int(x_expanded + w_expanded),
            int(y_expanded + h_expanded)
        ))
        
        # 应用变换
        if self.transform:
            try:
                full_image = self.transform(full_image)
                local_image = self.transform(local_image)
            except Exception as e:
                print(f"警告: 图像变换失败 {sample['filename']}: {e}")
                return None
        
        label = sample['label']
        
        return {
            'full_image': full_image,
            'local_image': local_image,
            'label': label,
            'filename': sample['filename']
        }


class SwinBackbone(nn.Module):
    """Swin Transformer Backbone"""
    
    def __init__(self, model_name='microsoft/swin-tiny-patch4-window7-224'):
        super().__init__()
        config = SwinConfig.from_pretrained(model_name)
        self.swin = SwinModel.from_pretrained(model_name, config=config)
        # Swin-tiny的embed_dim是96，但需要检查实际输出维度
        self.feature_dim = config.embed_dim * (config.depths[-1] // config.num_layers) if hasattr(config, 'depths') else config.embed_dim
        
        # 实际测试输出维度
        # SwinModel的输出是BaseModelOutput，last_hidden_state的形状是 [batch, num_patches, embed_dim]
        # 需要全局平均池化得到 [batch, embed_dim]
        # 对于swin-tiny: embed_dim = 96
        
    def forward(self, x):
        outputs = self.swin(x)
        # SwinModel返回BaseModelOutput，包含last_hidden_state
        # last_hidden_state形状: [batch_size, num_patches, embed_dim]
        # 需要全局平均池化: [batch_size, embed_dim]
        if hasattr(outputs, 'last_hidden_state'):
            # 对序列维度（num_patches）进行平均池化
            features = outputs.last_hidden_state.mean(dim=1)  # [batch, embed_dim]
            return features
        elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            # 备用方案：如果都没有，尝试获取第一个输出
            if isinstance(outputs, tuple) and len(outputs) > 0:
                return outputs[0].mean(dim=1)
            else:
                raise ValueError("无法从SwinModel获取特征")


class DualPathModel(nn.Module):
    """双路径模型：全局路径 + 局部路径"""
    
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        
        # 共享的Swin Backbone
        self.global_backbone = SwinBackbone()
        self.local_backbone = SwinBackbone()
        
        # 动态获取特征维度（通过前向传播测试）
        # 创建临时输入来测试特征维度
        # 注意：需要在eval模式下测试，避免batch norm等问题
        self.global_backbone.eval()
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            try:
                dummy_features = self.global_backbone(dummy_input)
                feature_dim = dummy_features.shape[1]  # [batch, feature_dim]
                print(f"检测到的特征维度: {feature_dim}")
            except Exception as e:
                print(f"警告: 无法自动检测特征维度: {e}")
                # 使用默认值（swin-tiny的embed_dim是96）
                feature_dim = 96
                print(f"使用默认特征维度: {feature_dim}")
        self.global_backbone.train()
        
        # 分类头
        self.global_classifier = nn.Linear(feature_dim, num_classes)
        self.local_classifier = nn.Linear(feature_dim, num_classes)
        
    def forward(self, full_image, local_image):
        # 全局路径
        global_features = self.global_backbone(full_image)
        global_logits = self.global_classifier(global_features)
        
        # 局部路径
        local_features = self.local_backbone(local_image)
        local_logits = self.local_classifier(local_features)
        
        return global_logits, local_logits, global_features, local_features


def load_all_json_files(data_dir: str) -> List[str]:
    """加载所有JSON标注文件（自动搜索，Kaggle适配）"""
    json_files = []
    
    # 搜索路径列表（Kaggle适配）
    search_paths = [
        data_dir,
        BASE_DIR,
        WORKING_DIR,
    ]
    
    # 添加所有输入数据集路径
    if IN_KAGGLE:
        for dataset_name in os.listdir(INPUT_DIR):
            dataset_path = os.path.join(INPUT_DIR, dataset_name)
            if os.path.isdir(dataset_path):
                search_paths.append(dataset_path)
    
    # 首先尝试直接路径
    for cls in CLASSES:
        found = False
        for search_path in search_paths:
            json_path = os.path.join(search_path, f'{cls}.json')
            if os.path.exists(json_path):
                json_files.append(json_path)
                found = True
                break
        
        if not found:
            # 递归搜索
            for search_path in search_paths:
                if os.path.exists(search_path):
                    pattern = os.path.join(search_path, '**', f'{cls}.json')
                    matches = glob.glob(pattern, recursive=True)
                    if matches:
                        json_files.append(matches[0])
                        found = True
                        break
        
        if not found:
            print(f"警告: 未找到 {cls}.json")
    
    # 如果找到了文件，显示位置
    if json_files:
        print(f"\n找到 {len(json_files)} 个JSON文件:")
        for f in json_files:
            print(f"  - {f}")
        
        # 检查是否所有文件都在同一目录
        dirs = [os.path.dirname(f) for f in json_files]
        if len(set(dirs)) == 1:
            print(f"\n所有文件在同一目录: {dirs[0]}")
    
    return json_files


def compute_pairwise_loss(global_logits, local_logits, labels, 
                          global_features=None, local_features=None,
                          consistency_weight=0.5, feature_align_weight=0.1):
    """
    计算成对损失函数
    
    Args:
        global_logits: 全局路径的logits [batch, num_classes]
        local_logits: 局部路径的logits [batch, num_classes]
        labels: 真实标签 [batch]
        global_features: 全局特征 [batch, feature_dim] (可选)
        local_features: 局部特征 [batch, feature_dim] (可选)
        consistency_weight: 一致性损失权重
        feature_align_weight: 特征对齐损失权重
    
    Returns:
        total_loss: 总损失
        loss_dict: 损失字典
    """
    # 分类损失（交叉熵）
    global_ce = F.cross_entropy(global_logits, labels)
    local_ce = F.cross_entropy(local_logits, labels)
    classification_loss = global_ce + local_ce
    
    # 一致性损失（KL散度）
    global_probs = F.softmax(global_logits, dim=1)
    local_probs = F.softmax(local_logits, dim=1)
    
    # KL(global || local) + KL(local || global)
    consistency_loss = F.kl_div(
        F.log_softmax(global_logits, dim=1), 
        local_probs, 
        reduction='batchmean'
    ) + F.kl_div(
        F.log_softmax(local_logits, dim=1), 
        global_probs, 
        reduction='batchmean'
    )
    
    # 特征对齐损失（如果提供了特征）
    feature_align_loss = torch.tensor(0.0, device=global_logits.device)
    if global_features is not None and local_features is not None:
        # L2距离
        feature_align_loss = F.mse_loss(global_features, local_features)
    
    # 总损失
    total_loss = (
        classification_loss + 
        consistency_weight * consistency_loss + 
        feature_align_weight * feature_align_loss
    )
    
    loss_dict = {
        'total': total_loss.item(),
        'classification': classification_loss.item(),
        'global_ce': global_ce.item(),
        'local_ce': local_ce.item(),
        'consistency': consistency_loss.item(),
        'feature_align': feature_align_loss.item()
    }
    
    return total_loss, loss_dict


def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda',
                consistency_weight=0.5, feature_align_weight=0.1):
    """
    训练双路径模型（使用成对损失函数）
    
    Args:
        model: 双路径模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        device: 设备
        consistency_weight: 一致性损失权重
        feature_align_weight: 特征对齐损失权重
    """
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss_dict = {
            'total': 0.0,
            'classification': 0.0,
            'global_ce': 0.0,
            'local_ce': 0.0,
            'consistency': 0.0,
            'feature_align': 0.0
        }
        
        for batch in train_loader:
            # 跳过None样本
            if batch is None:
                continue
            
            full_images = batch['full_image'].to(device)
            local_images = batch['local_image'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播（获取logits和features）
            global_logits, local_logits, global_features, local_features = model(
                full_images, local_images
            )
            
            # 计算成对损失
            total_loss, loss_dict = compute_pairwise_loss(
                global_logits, local_logits, labels,
                global_features, local_features,
                consistency_weight=consistency_weight,
                feature_align_weight=feature_align_weight
            )
            
            # 反向传播
            total_loss.backward()
            optimizer.step()
            
            # 累计损失
            for key in train_loss_dict:
                train_loss_dict[key] += loss_dict[key]
        
        # 验证阶段
        model.eval()
        val_loss_dict = {
            'total': 0.0,
            'classification': 0.0,
            'global_ce': 0.0,
            'local_ce': 0.0,
            'consistency': 0.0,
            'feature_align': 0.0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                # 跳过None样本
                if batch is None:
                    continue
                
                full_images = batch['full_image'].to(device)
                local_images = batch['local_image'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                global_logits, local_logits, global_features, local_features = model(
                    full_images, local_images
                )
                
                # 计算损失
                _, loss_dict = compute_pairwise_loss(
                    global_logits, local_logits, labels,
                    global_features, local_features,
                    consistency_weight=consistency_weight,
                    feature_align_weight=feature_align_weight
                )
                
                for key in val_loss_dict:
                    val_loss_dict[key] += loss_dict[key]
        
        # 计算平均损失
        num_batches = len(train_loader)
        for key in train_loss_dict:
            train_loss_dict[key] /= num_batches
        
        num_val_batches = len(val_loader)
        for key in val_loss_dict:
            val_loss_dict[key] /= num_val_batches
        
        train_losses.append(train_loss_dict['total'])
        val_losses.append(val_loss_dict['total'])
        
        # 打印损失信息
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'  Train Loss: {train_loss_dict["total"]:.4f}')
        print(f'    - Classification: {train_loss_dict["classification"]:.4f} '
              f'(Global CE: {train_loss_dict["global_ce"]:.4f}, '
              f'Local CE: {train_loss_dict["local_ce"]:.4f})')
        print(f'    - Consistency: {train_loss_dict["consistency"]:.4f}')
        print(f'    - Feature Align: {train_loss_dict["feature_align"]:.4f}')
        print(f'  Val Loss: {val_loss_dict["total"]:.4f}')
        print(f'    - Classification: {val_loss_dict["classification"]:.4f} '
              f'(Global CE: {val_loss_dict["global_ce"]:.4f}, '
              f'Local CE: {val_loss_dict["local_ce"]:.4f})')
        print(f'    - Consistency: {val_loss_dict["consistency"]:.4f}')
        print(f'    - Feature Align: {val_loss_dict["feature_align"]:.4f}')
    
    return train_losses, val_losses


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("双路径鱼类分类模型 - Kaggle版本")
    print("=" * 60)
    
    # Kaggle环境下的数据目录查找
    if IN_KAGGLE:
        print("\n🔍 在Kaggle环境中查找数据...")
        
        # 列出所有输入数据集
        print(f"\n输入数据集列表 ({INPUT_DIR}):")
        datasets = []
        for item in os.listdir(INPUT_DIR):
            item_path = os.path.join(INPUT_DIR, item)
            if os.path.isdir(item_path):
                datasets.append(item)
                print(f"  📁 {item}")
        
        # 尝试在每个数据集中查找JSON文件
        data_dir = None
        for dataset_name in datasets:
            dataset_path = os.path.join(INPUT_DIR, dataset_name)
            json_test = load_all_json_files(dataset_path)
            if json_test:
                data_dir = dataset_path
                print(f"\n✅ 在数据集 '{dataset_name}' 中找到JSON文件")
                break
        
        if data_dir is None:
            # 尝试在工作目录查找
            print("\n在工作目录中查找...")
            json_test = load_all_json_files(WORKING_DIR)
            if json_test:
                data_dir = WORKING_DIR
            else:
                print("⚠️  未找到JSON文件，请确保数据集已添加到Notebook Input")
                print("\n提示:")
                print("1. 在Kaggle Notebook中，点击右侧 'Add Input'")
                print("2. 选择包含JSON文件的数据集")
                print("3. JSON文件应该在 /kaggle/input/dataset-name/ 目录下")
                return
    else:
        # 本地环境
        data_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"\n本地环境，使用目录: {data_dir}")
    
    # 分析数据集（自动搜索JSON文件）
    json_files = load_all_json_files(data_dir)
    if not json_files:
        print("\n❌ 错误: 未找到任何JSON文件！")
        if IN_KAGGLE:
            print("\n请确保:")
            print("1. 数据集已添加到Notebook的Input中")
            print("2. JSON文件在数据集的根目录或子目录中")
            print("3. JSON文件命名正确（ALB.json, BET.json等）")
        return
    
    print(f"\n✅ 找到 {len(json_files)} 个JSON文件")
    
    # 定义图像变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    print("\n" + "=" * 60)
    print("创建数据集")
    print("=" * 60)
    
    # 创建数据集，跳过缺失的图片
    full_dataset = FishDataset(
        data_dir, 
        json_files, 
        transform=transform, 
        expand_ratio=0.2,
        skip_missing=True  # 跳过缺失的图片
    )
    
    print(f"数据集大小: {len(full_dataset)} 个样本")
    if len(full_dataset) == 0:
        print("\n⚠️  警告: 所有图片都找不到！")
        if IN_KAGGLE:
            print("请检查:")
            print("1. 图片文件是否在数据集中")
            print("2. 图片路径是否与JSON中的路径匹配")
            print("3. 图片文件格式是否正确")
        return
    
    # 划分训练集和验证集
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建数据加载器
    batch_size = 16
    num_workers = 0  # Kaggle中建议使用0
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # 创建模型
    print("\n" + "=" * 60)
    print("创建双路径模型")
    print("=" * 60)
    print("架构:")
    print("  全局路径: Full Image -> Swin Backbone -> Global Feature -> Classification -> CE(global)")
    print("  局部路径: BBox Crop Image (expand 20%) -> Swin Backbone -> Local Feature -> Classification -> CE(local)")
    print("  最终损失: Final Loss = CE(global) + CE(local) + Consistency + Feature Align")
    
    # 自动检测设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = DualPathModel(num_classes=NUM_CLASSES).to(device)
    
    # 训练模型
    print("\n" + "=" * 60)
    print("开始训练（使用成对损失函数）")
    print("=" * 60)
    print("损失函数组成:")
    print("  - 分类损失: CE(global) + CE(local)")
    print("  - 一致性损失: KL散度确保全局和局部预测一致")
    print("  - 特征对齐损失: 让全局和局部特征在特征空间中更接近")
    print("  - 最终损失 = 分类损失 + 0.5 * 一致性损失 + 0.1 * 特征对齐损失")
    
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=10, device=device,
        consistency_weight=0.5,  # 一致性损失权重
        feature_align_weight=0.1  # 特征对齐损失权重
    )
    
    # 保存模型（保存到工作目录）
    model_path = os.path.join(WORKING_DIR, 'dual_path_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ 模型已保存到: {model_path}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Dual Path Model Training Loss')
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(WORKING_DIR, 'training_loss.png')
    plt.savefig(loss_plot_path)
    print(f"✅ 训练损失曲线已保存到: {loss_plot_path}")
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    if IN_KAGGLE:
        print(f"\n📁 输出文件位置:")
        print(f"  - 模型: {model_path}")
        print(f"  - 损失曲线: {loss_plot_path}")
        print("\n💡 提示: 这些文件会自动保存到Kaggle Notebook的输出中")


if __name__ == '__main__':
    main()

