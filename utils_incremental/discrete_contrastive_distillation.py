"""
Discrete Contrastive Distillation (DCD) Module
离散对比蒸馏模块

实现方案4（离散对比蒸馏）+ 方案1（类别特征选择）

核心功能：
1. 类别特征选择：为每个类别识别Top-K重要特征维度
2. 离散激活：保留每个样本的Top-K强特征维度
3. 对比蒸馏：旧类强约束，新类弱约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiscreteContrastiveDistillation(nn.Module):
    """
    离散对比蒸馏类
    
    功能：
    - 计算类别特征重要性mask
    - 执行离散激活
    - 执行对比蒸馏损失计算
    """
    
    def __init__(self, 
                 feature_dim=512, 
                 top_k_class=50, 
                 top_k_sample=50,
                 alpha=0.01,
                 temperature=0.1,
                 old_class_weight=1.0,
                 new_class_weight=0.3,
                 importance_method='combined'):
        """
        Args:
            feature_dim: 特征维度（512）
            top_k_class: 每个类别保留的重要维度数量
            top_k_sample: 每个样本保留的强激活维度数量
            alpha: 弱特征的保留系数（leaky）
            temperature: 相似度计算的温度参数
            old_class_weight: 旧类蒸馏损失权重
            new_class_weight: 新类蒸馏损失权重
            importance_method: 重要性计算方法 ('strength', 'frequency', 'combined')
        """
        super(DiscreteContrastiveDistillation, self).__init__()
        
        self.feature_dim = feature_dim
        self.top_k_class = top_k_class
        self.top_k_sample = top_k_sample
        self.alpha = alpha
        self.temperature = temperature
        self.old_class_weight = old_class_weight
        self.new_class_weight = new_class_weight
        self.importance_method = importance_method
        
        # 存储每个类别的重要性mask
        self.class_importance_masks = {}
        self.class_importance_indices = {}
        
        print(f"\n[DCD] Discrete Contrastive Distillation initialized:")
        print(f"  Feature dim: {feature_dim}")
        print(f"  Top-K per class: {top_k_class}")
        print(f"  Top-K per sample: {top_k_sample}")
        print(f"  Alpha (leaky): {alpha}")
        print(f"  Temperature: {temperature}")
        print(f"  Old class weight: {old_class_weight}")
        print(f"  New class weight: {new_class_weight}")
        
    def compute_class_importance_masks(self, model, dataloader, num_classes, device):
        """
        计算每个类别的特征重要性mask
        
        在任务开始时调用一次，使用ref_model和当前任务的有标签数据
        
        Args:
            model: 模型（通常是ref_model）
            dataloader: 数据加载器（有标签数据）
            num_classes: 类别总数
            device: 设备
        """
        print(f"\n{'='*80}")
        print(f"[DCD] Computing class importance masks...")
        print(f"  Total classes: {num_classes}")
        print(f"  Method: {self.importance_method}")
        
        model.eval()
        
        # 收集每个类别的特征
        class_features_list = {c: [] for c in range(num_classes)}
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # 兼容不同的dataloader格式
                if len(batch_data) == 6:  # (indexs, inputs, inputs_s, targets, flags, on_flags)
                    _, inputs, _, targets, _, _ = batch_data
                elif len(batch_data) == 2:  # (inputs, targets)
                    inputs, targets = batch_data
                else:
                    continue
                
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 提取特征 - 兼容DER和非DER版本
                features = None
                try:
                    # 尝试DER版本的调用方式（返回字典）
                    outs = model(inputs)
                    if isinstance(outs, dict):
                        features = outs['con_feats']
                        if batch_idx == 0:
                            print(f"  [DEBUG] DER model detected, con_feats shape: {features.shape}")
                    else:
                        # 非DER版本：model(inputs) 只返回logits，需要return_feats=True
                        if batch_idx == 0:
                            print(f"  [DEBUG] Non-DER model detected (outs type: {type(outs)}), trying return_feats=True...")
                        raise ValueError("Need return_feats=True for non-DER model")
                except Exception as e:
                    # 非DER版本必须使用return_feats=True
                    try:
                        outs = model(inputs, return_feats=True)
                        if isinstance(outs, tuple) and len(outs) >= 3:
                            features = outs[2]  # (logits, feats, con_feats, non_feats)
                            if batch_idx == 0:
                                print(f"  [DEBUG] Non-DER with return_feats=True, con_feats shape: {features.shape}")
                        else:
                            if batch_idx == 0:
                                print(f"  [ERROR] Unexpected output format: {type(outs)}, len: {len(outs) if isinstance(outs, tuple) else 'N/A'}")
                            continue
                    except Exception as e2:
                        if batch_idx == 0:
                            print(f"  [ERROR] Both methods failed!")
                            print(f"    Error 1: {e}")
                            print(f"    Error 2: {e2}")
                        continue
                
                if features is None:
                    continue
                
                # 按类别收集特征
                for c in range(num_classes):
                    mask = (targets == c)
                    if mask.sum() > 0:
                        class_features_list[c].append(features[mask].cpu())
        
        # 打印特征收集统计
        print(f"\n{'='*80}")
        print(f"[DCD] Feature collection summary:")
        for c in range(num_classes):
            if len(class_features_list[c]) > 0:
                total_samples = sum(f.shape[0] for f in class_features_list[c])
                first_shape = class_features_list[c][0].shape
                print(f"  Class {c:2d}: {total_samples:4d} samples, first batch shape: {first_shape}")
            else:
                print(f"  Class {c:2d}: 0 samples ⚠️ NO DATA")
        print(f"{'='*80}\n")
        
        # 为每个类别计算重要性并创建mask
        for c in range(num_classes):
            # 检查是否有样本
            if len(class_features_list[c]) == 0:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"ERROR: Class {c} has NO samples!\n"
                    f"{'='*80}\n"
                    f"This class has no data for computing importance mask.\n"
                    f"Possible causes:\n"
                    f"  1. Dataloader doesn't contain samples for class {c}\n"
                    f"  2. Model feature extraction failed for this class\n"
                    f"  3. Targets are not correctly assigned\n"
                    f"\nPlease check the dataloader and ensure all classes in [0, {num_classes-1}] have data.\n"
                    f"{'='*80}\n"
                )
            
            # 合并该类所有特征
            all_feats = torch.cat(class_features_list[c], dim=0)  # [n_samples, feature_dim]
            n_samples = all_feats.shape[0]
            
            # 检查特征形状
            if all_feats.dim() != 2:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"ERROR: Class {c} has invalid feature shape!\n"
                    f"{'='*80}\n"
                    f"  Got shape: {all_feats.shape} ({all_feats.dim()}D tensor)\n"
                    f"  Expected: [n_samples, {self.feature_dim}] (2D tensor)\n"
                    f"\nThe model's con_feats output has incorrect dimensions.\n"
                    f"{'='*80}\n"
                )
            
            actual_dim = all_feats.shape[1]
            if actual_dim != self.feature_dim:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"ERROR: Class {c} feature dimension mismatch!\n"
                    f"{'='*80}\n"
                    f"  Got feature dim: {actual_dim}\n"
                    f"  Expected dim: {self.feature_dim}\n"
                    f"  Feature shape: {all_feats.shape}\n"
                    f"\nThe model's con_feats output has wrong dimension.\n"
                    f"Check if you're using the correct model version (DER vs non-DER).\n"
                    f"{'='*80}\n"
                )
            
            # 计算重要性分数
            if self.importance_method == 'strength':
                importance = all_feats.abs().mean(dim=0)
            elif self.importance_method == 'frequency':
                threshold = all_feats.abs().mean()
                importance = (all_feats.abs() > threshold).float().mean(dim=0)
            elif self.importance_method == 'combined':
                strength = all_feats.abs().mean(dim=0)
                threshold = all_feats.abs().mean()
                frequency = (all_feats.abs() > threshold).float().mean(dim=0)
                importance = 0.6 * strength + 0.4 * frequency
            else:
                raise ValueError(f"Unknown importance method: {self.importance_method}")
            
            # 检查importance向量
            if importance.numel() != self.feature_dim:
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"ERROR: Class {c} importance vector has wrong size!\n"
                    f"{'='*80}\n"
                    f"  Importance size: {importance.numel()}\n"
                    f"  Expected size: {self.feature_dim}\n"
                    f"  Importance shape: {importance.shape}\n"
                    f"\nThis is an internal error in importance calculation.\n"
                    f"{'='*80}\n"
                )
            
            # 选择Top-K维度
            top_k = min(self.top_k_class, self.feature_dim)
            
            # 再次检查（防御性编程）
            if top_k > importance.numel():
                raise RuntimeError(
                    f"\n{'='*80}\n"
                    f"ERROR: Class {c} cannot select top-{top_k} dimensions!\n"
                    f"{'='*80}\n"
                    f"  Importance vector size: {importance.numel()}\n"
                    f"  Requested top_k: {top_k}\n"
                    f"  top_k_class setting: {self.top_k_class}\n"
                    f"  feature_dim setting: {self.feature_dim}\n"
                    f"\nThis should never happen. Please report this bug.\n"
                    f"{'='*80}\n"
                )
            
            top_values, top_indices = torch.topk(importance, k=top_k)
            
            # 创建mask（0/1）
            mask = torch.zeros(self.feature_dim)
            mask[top_indices] = 1.0
            
            self.class_importance_masks[c] = mask
            self.class_importance_indices[c] = top_indices
            
            print(f"  ✓ Class {c:2d}: {n_samples:4d} samples, top-{top_k}/{self.feature_dim} dims | "
                  f"Importance [{importance.min():.4f}, {importance.max():.4f}]")
        
        print(f"[DCD] Class importance masks computed successfully!")
        print(f"{'='*80}\n")
    
    def discrete_activation(self, features):
        """
        离散激活：保留每个样本的Top-K强特征
        
        Args:
            features: [batch_size, feature_dim]
            
        Returns:
            features_discrete: [batch_size, feature_dim] 离散激活后的特征
        """
        batch_size = features.shape[0]
        
        # 找到每个样本的Top-K维度
        abs_features = features.abs()
        top_k = min(self.top_k_sample, self.feature_dim)
        topk_values, topk_indices = torch.topk(abs_features, k=top_k, dim=1)
        
        # 创建mask
        mask = torch.zeros_like(features)
        for i in range(batch_size):
            mask[i, topk_indices[i]] = 1.0
        
        # 应用离散激活：强特征保持，弱特征削弱到alpha倍
        features_discrete = features * mask + features * (1 - mask) * self.alpha
        
        return features_discrete
    
    def apply_class_masks_to_prototypes(self, prototypes, num_classes):
        """
        将类别特征mask应用到原型上
        
        Args:
            prototypes: [num_classes, feature_dim]
            num_classes: 类别数量
            
        Returns:
            prototypes_masked: [num_classes, feature_dim] mask后的原型
        """
        prototypes_masked = []
        
        for c in range(num_classes):
            if c in self.class_importance_masks:
                mask = self.class_importance_masks[c].to(prototypes.device)
                proto_masked = prototypes[c] * mask
            else:
                # 如果没有mask，使用原始原型
                proto_masked = prototypes[c]
            
            prototypes_masked.append(proto_masked)
        
        return torch.stack(prototypes_masked, dim=0)
    
    def discrete_contrastive_distillation_loss(self,
                                              feat_student,
                                              feat_teacher,
                                              prototypes_ref,
                                              old_cn,
                                              total_cn):
        """
        计算离散对比蒸馏损失
        
        Args:
            feat_student: Student特征 [batch_size, feature_dim]
            feat_teacher: Teacher特征 [batch_size, feature_dim]
            prototypes_ref: 原型 [total_cn, feature_dim]
            old_cn: 旧类数量
            total_cn: 总类数量
            
        Returns:
            loss: 蒸馏损失
            stats: 统计信息字典
        """
        # Step 1: 离散激活
        feat_student_discrete = self.discrete_activation(feat_student)
        feat_teacher_discrete = self.discrete_activation(feat_teacher)
        
        # Step 2: 归一化
        feat_student_norm = F.normalize(feat_student_discrete, p=2, dim=1)
        feat_teacher_norm = F.normalize(feat_teacher_discrete, p=2, dim=1)
        
        # Step 3: 应用类别mask到原型
        prototypes_masked = self.apply_class_masks_to_prototypes(prototypes_ref, total_cn)
        prototypes_norm = F.normalize(prototypes_masked, p=2, dim=1)
        
        # Step 4: 计算相似度
        sim_teacher = feat_teacher_norm @ prototypes_norm.T / self.temperature  # [batch, total_cn]
        sim_student = feat_student_norm @ prototypes_norm.T / self.temperature  # [batch, total_cn]
        
        # Step 5: 转换为概率分布
        teacher_prob = F.softmax(sim_teacher, dim=1)
        student_log_prob = F.log_softmax(sim_student, dim=1)
        
        # Step 6: 分离旧类和新类，计算对比蒸馏损失
        if old_cn > 0:
            # 旧类：强约束
            loss_old = F.kl_div(
                student_log_prob[:, :old_cn],
                teacher_prob[:, :old_cn],
                reduction='batchmean'
            ) * self.old_class_weight
        else:
            loss_old = torch.tensor(0.0).to(feat_student.device)
        
        if total_cn > old_cn:
            # 新类：弱约束
            loss_new = F.kl_div(
                student_log_prob[:, old_cn:],
                teacher_prob[:, old_cn:],
                reduction='batchmean'
            ) * self.new_class_weight
        else:
            loss_new = torch.tensor(0.0).to(feat_student.device)
        
        # 总损失
        loss_total = loss_old + loss_new
        
        # 统计信息
        stats = {
            'loss_old': loss_old.item(),
            'loss_new': loss_new.item(),
            'loss_total': loss_total.item(),
            'teacher_conf_mean': teacher_prob.max(dim=1)[0].mean().item(),
            'student_conf_mean': F.softmax(sim_student, dim=1).max(dim=1)[0].mean().item(),
        }
        
        return loss_total, stats
    
    def get_stats_string(self, stats):
        """
        获取统计信息的格式化字符串
        """
        return (f"DCD Loss: Total={stats['loss_total']:.4f} "
                f"(Old={stats['loss_old']:.4f}, New={stats['loss_new']:.4f}), "
                f"T_conf={stats['teacher_conf_mean']:.3f}, "
                f"S_conf={stats['student_conf_mean']:.3f}")


def create_dcd_module(args, device):
    """
    创建离散对比蒸馏模块的工厂函数
    
    Args:
        args: 命令行参数
        device: 设备
        
    Returns:
        dcd_module: DiscreteContrastiveDistillation实例，如果未启用则返回None
    """
    if not getattr(args, 'enable_dcd', False):
        return None
    
    dcd_module = DiscreteContrastiveDistillation(
        feature_dim=args.dim,
        top_k_class=getattr(args, 'dcd_top_k_class', 50),
        top_k_sample=getattr(args, 'dcd_top_k_sample', 50),
        alpha=getattr(args, 'dcd_alpha', 0.01),
        temperature=getattr(args, 'dcd_temperature', 0.1),
        old_class_weight=getattr(args, 'dcd_old_weight', 1.0),
        new_class_weight=getattr(args, 'dcd_new_weight', 0.3),
        importance_method=getattr(args, 'dcd_importance_method', 'combined')
    ).to(device)
    
    return dcd_module
