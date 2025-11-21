"""
测试成对损失函数
验证损失函数是否正确计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dual_path_analysis import compute_pairwise_loss

def test_pairwise_loss():
    """测试成对损失函数"""
    print("=" * 60)
    print("测试成对损失函数")
    print("=" * 60)
    
    # 创建模拟数据
    batch_size = 4
    num_classes = 7
    feature_dim = 768
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 模拟logits
    global_logits = torch.randn(batch_size, num_classes).to(device)
    local_logits = torch.randn(batch_size, num_classes).to(device)
    
    # 模拟标签
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)
    
    # 模拟特征
    global_features = torch.randn(batch_size, feature_dim).to(device)
    local_features = torch.randn(batch_size, feature_dim).to(device)
    
    print(f"\n输入形状:")
    print(f"  Global logits: {global_logits.shape}")
    print(f"  Local logits: {local_logits.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Global features: {global_features.shape}")
    print(f"  Local features: {local_features.shape}")
    
    # 测试1: 只有分类损失
    print("\n" + "-" * 60)
    print("测试1: 只有分类损失（consistency_weight=0, feature_align_weight=0）")
    print("-" * 60)
    loss1, loss_dict1 = compute_pairwise_loss(
        global_logits, local_logits, labels,
        consistency_weight=0.0,
        feature_align_weight=0.0
    )
    print(f"总损失: {loss_dict1['total']:.4f}")
    print(f"  分类损失: {loss_dict1['classification']:.4f}")
    print(f"    全局CE: {loss_dict1['global_ce']:.4f}")
    print(f"    局部CE: {loss_dict1['local_ce']:.4f}")
    print(f"  一致性损失: {loss_dict1['consistency']:.4f}")
    print(f"  特征对齐损失: {loss_dict1['feature_align']:.4f}")
    
    # 测试2: 包含一致性损失
    print("\n" + "-" * 60)
    print("测试2: 包含一致性损失（consistency_weight=0.5, feature_align_weight=0）")
    print("-" * 60)
    loss2, loss_dict2 = compute_pairwise_loss(
        global_logits, local_logits, labels,
        global_features=None,
        local_features=None,
        consistency_weight=0.5,
        feature_align_weight=0.0
    )
    print(f"总损失: {loss_dict2['total']:.4f}")
    print(f"  分类损失: {loss_dict2['classification']:.4f}")
    print(f"  一致性损失: {loss_dict2['consistency']:.4f} (权重: 0.5)")
    print(f"  特征对齐损失: {loss_dict2['feature_align']:.4f}")
    
    # 测试3: 完整损失（包含所有组件）
    print("\n" + "-" * 60)
    print("测试3: 完整损失（包含所有组件）")
    print("-" * 60)
    loss3, loss_dict3 = compute_pairwise_loss(
        global_logits, local_logits, labels,
        global_features=global_features,
        local_features=local_features,
        consistency_weight=0.5,
        feature_align_weight=0.1
    )
    print(f"总损失: {loss_dict3['total']:.4f}")
    print(f"  分类损失: {loss_dict3['classification']:.4f}")
    print(f"  一致性损失: {loss_dict3['consistency']:.4f} (权重: 0.5)")
    print(f"  特征对齐损失: {loss_dict3['feature_align']:.4f} (权重: 0.1)")
    
    # 验证损失计算
    expected_total = (
        loss_dict3['classification'] +
        0.5 * loss_dict3['consistency'] +
        0.1 * loss_dict3['feature_align']
    )
    print(f"\n验证:")
    print(f"  计算的总损失: {loss_dict3['total']:.4f}")
    print(f"  期望的总损失: {expected_total:.4f}")
    print(f"  差异: {abs(loss_dict3['total'] - expected_total):.6f}")
    
    if abs(loss_dict3['total'] - expected_total) < 1e-4:
        print("  ✓ 损失计算正确！")
    else:
        print("  ✗ 损失计算有误！")
    
    # 测试4: 测试成对约束（当全局和局部预测一致时，一致性损失应该较小）
    print("\n" + "-" * 60)
    print("测试4: 成对约束测试（相同logits应该有一致性损失≈0）")
    print("-" * 60)
    identical_logits = global_logits.clone()
    loss4, loss_dict4 = compute_pairwise_loss(
        identical_logits, identical_logits, labels,
        global_features=global_features,
        local_features=local_features,
        consistency_weight=0.5,
        feature_align_weight=0.1
    )
    print(f"当全局和局部logits相同时:")
    print(f"  一致性损失: {loss_dict4['consistency']:.6f}")
    if loss_dict4['consistency'] < 1e-5:
        print("  ✓ 一致性损失接近0，符合预期！")
    else:
        print(f"  ⚠ 一致性损失为 {loss_dict4['consistency']:.6f}，可能由于数值精度问题")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == '__main__':
    test_pairwise_loss()



