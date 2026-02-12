#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COVID-19 新增病例百分比预测 - 修复版本
Kaggle Grandmaster 级别优化方案
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class EnhancedCOVID19Predictor:
    def __init__(self, device: str = None, use_advanced_features: bool = True):
        """
        初始化增强版预测器
        
        Args:
            device: 计算设备 (cpu/cuda)
            use_advanced_features: 是否使用高级特征工程
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_advanced_features = use_advanced_features
        
        # 特征工程组件
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features_mask = None
        
        # 模型
        self.model = None
        self.feature_columns = None
        
        print(f"使用设备: {self.device}")
        
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加载数据
        """
        print("正在加载数据...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"训练集形状: {train_df.shape}")
        print(f"测试集形状: {test_df.shape}")
        
        # 检查目标变量分布
        if 'tested_positive_day3' in train_df.columns:
            target_stats = train_df['tested_positive_day3'].describe()
            print(f"\n目标变量统计:")
            print(f"均值: {target_stats['mean']:.4f}")
            print(f"标准差: {target_stats['std']:.4f}")
            print(f"最小值: {target_stats['min']:.4f}")
            print(f"25%分位数: {target_stats['25%']:.4f}")
            print(f"中位数: {target_stats['50%']:.4f}")
            print(f"75%分位数: {target_stats['75%']:.4f}")
            print(f"最大值: {target_stats['max']:.4f}")
        
        return train_df, test_df
    
    def create_advanced_features(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        创建高级特征工程 - 简化版，避免特征过多
        """
        df = df.copy()
        
        # 基础特征：识别州列
        state_cols = [col for col in df.columns if len(col) == 2 and col.isupper()]
        
        # 交互特征 - 简化版本
        interaction_groups = [
            ('cli_day3', 'wearing_mask_7d_day3'),
            ('ili_day3', 'wearing_mask_7d_day3'),
            ('cli_day3', 'wbelief_masking_effective_day3'),
            ('wearing_mask_7d_day3', 'wcovid_vaccinated_friends_day3'),
        ]
        
        for feat1, feat2 in interaction_groups:
            if feat1 in df.columns and feat2 in df.columns:
                df[f"{feat1.split('_')[0]}_{feat2.split('_')[0]}_product"] = df[feat1] * df[feat2]
        
        # 趋势特征
        for feature_base in ['cli', 'ili', 'tested_positive', 'wearing_mask_7d']:
            day1_col = f"{feature_base}_day1"
            day3_col = f"{feature_base}_day3"
            
            if day1_col in df.columns and day3_col in df.columns:
                df[f"{feature_base}_trend"] = df[day3_col] - df[day1_col]
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False, 
                         fit_selector: bool = False, y: np.ndarray = None) -> np.ndarray:
        """
        准备特征
        """
        df = df.copy()
        
        # 移除ID列和目标列
        columns_to_drop = ['id']
        if 'tested_positive_day3' in df.columns:
            columns_to_drop.append('tested_positive_day3')
        
        # 只移除存在的列
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        features_df = df.drop(columns=columns_to_drop)
        
        # 保存特征列名（只在训练阶段）
        if fit_scaler:
            self.feature_columns = features_df.columns.tolist()
        
        # 在测试阶段，确保列顺序与训练时一致
        if not fit_scaler and hasattr(self, 'feature_columns') and self.feature_columns is not None:
            # 检查是否有缺失的列
            missing_cols = set(self.feature_columns) - set(features_df.columns)
            if missing_cols:
                # 添加缺失的列（用0填充）
                for col in missing_cols:
                    features_df[col] = 0
            # 确保列顺序一致
            features_df = features_df[self.feature_columns]
        
        # 转换为numpy数组
        features = features_df.values
        
        # 处理无限值和NaN值
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 标准化特征
        if fit_scaler:
            features = self.scaler.fit_transform(features)
        else:
            # 确保scaler已经拟合
            if hasattr(self.scaler, 'mean_'):
                features = self.scaler.transform(features)
            else:
                raise ValueError("Scaler has not been fitted yet.")
        
        # 特征选择 - 使用SelectKBest
        if fit_selector and y is not None:
            # 使用SelectKBest选择最重要的特征
            self.feature_selector = SelectKBest(f_regression, k=min(15, features.shape[1]))
            features = self.feature_selector.fit_transform(features, y)
            self.selected_features_mask = self.feature_selector.get_support()
            print(f"选择了 {np.sum(self.selected_features_mask)} 个重要特征")
        elif hasattr(self, 'feature_selector') and self.feature_selector is not None:
            # 使用训练好的特征选择
            features = self.feature_selector.transform(features)
        
        return features
    
    def create_dataset(self, train_df: pd.DataFrame, val_ratio: float = 0.2) -> Tuple:
        """
        创建训练和验证数据集（时间序列顺序）
        """
        # 先进行高级特征工程（训练阶段）
        if self.use_advanced_features:
            train_df_processed = self.create_advanced_features(train_df, is_train=True)
        else:
            train_df_processed = train_df.copy()
        
        print(f"特征工程后训练集形状: {train_df_processed.shape}")
        
        # 计算分割点
        split_idx = int(len(train_df_processed) * (1 - val_ratio))
        
        # 分割训练集和验证集（不shuffle！）
        train_data = train_df_processed.iloc[:split_idx].copy()
        val_data = train_df_processed.iloc[split_idx:].copy()
        
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")
        
        return train_data, val_data
    
    def build_enhanced_model(self, input_dim: int) -> nn.Module:
        """
        构建增强的神经网络模型 - 修复残差连接维度问题
        """
        class EnhancedCOVIDNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                
                self.input_norm = nn.BatchNorm1d(input_dim)
                
                # 更合理的网络结构
                self.block1 = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )
                
                self.block2 = nn.Sequential(
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(0.15)
                )
                
                self.block3 = nn.Sequential(
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                )
                
                # 输出层
                self.output_layer = nn.Sequential(
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1)
                )
                
                # 初始化权重
                self._init_weights()
            
            def _init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                # 输入归一化
                x_norm = self.input_norm(x)
                
                # 第一层
                x1 = self.block1(x_norm)
                
                # 第二层
                x2 = self.block2(x1)
                
                # 第三层
                x3 = self.block3(x2)
                
                # 输出
                output = self.output_layer(x3)
                
                return output.squeeze()
        
        return EnhancedCOVIDNet(input_dim).to(self.device)
    
    def train_epoch(self, model, train_loader, criterion, optimizer, scheduler, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        total_mae = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # 计算MAE
            mae = torch.mean(torch.abs(output - target))
            
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            total_mae += mae.item()
            
        avg_loss = total_loss / len(train_loader)
        avg_mae = total_mae / len(train_loader)
        
        return avg_loss, avg_mae
    
    def validate(self, model, val_loader, criterion):
        """验证模型"""
        model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                total_loss += loss.item()
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算多种指标
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        mae = np.mean(np.abs(all_preds - all_targets))
        mse = np.mean((all_preds - all_targets) ** 2)
        rmse = np.sqrt(mse)
        
        # R²分数
        ss_res = np.sum((all_targets - all_preds) ** 2)
        ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        return total_loss / len(val_loader), mae, rmse, r2, all_preds, all_targets
    
    def train(self, train_data, val_data, n_epochs: int = 200, 
              lr: float = 0.001, batch_size: int = 64):
        """
        训练模型
        """
        print("\n准备训练数据...")
        
        # 准备训练特征和目标
        y_train = train_data['tested_positive_day3'].values
        X_train = self.prepare_features(train_data, fit_scaler=True, fit_selector=True, y=y_train)
        
        # 准备验证特征和目标
        y_val = val_data['tested_positive_day3'].values
        X_val = self.prepare_features(val_data, fit_scaler=False, fit_selector=False)
        
        print(f"训练特征形状: {X_train.shape}")
        print(f"验证特征形状: {X_val.shape}")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # 创建数据集和数据加载器（注意：shuffle=False！）
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0, pin_memory=True)
        
        # 构建模型
        self.model = self.build_enhanced_model(X_train.shape[1])
        
        # 定义损失函数 - 使用MAE损失（对异常值更鲁棒）
        criterion = nn.L1Loss()
        
        # 定义优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-3)
        
        # 学习率调度器
        total_steps = n_epochs * len(train_loader)
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=lr,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        print("\n开始训练...")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        best_rmse = float('inf')
        best_r2 = -float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        val_rmses = []
        val_r2s = []
        
        for epoch in range(n_epochs):
            # 训练
            train_loss, train_mae = self.train_epoch(
                self.model, train_loader, criterion, optimizer, scheduler, epoch
            )
            
            # 验证
            val_loss, val_mae, val_rmse, val_r2, _, _ = self.validate(
                self.model, val_loader, criterion
            )
            
            # 记录指标
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_rmses.append(val_rmse)
            val_r2s.append(val_r2)
            
            # 保存最佳模型
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_r2 = val_r2
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                # 保存最佳模型
                torch.save(best_model_state, 'best_model.pth')
            else:
                patience_counter += 1
            
            # 定期打印进度
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{n_epochs}:")
                print(f"  Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
                print(f"  Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
                print(f"  Val RMSE: {val_rmse:.4f}, Val R²: {val_r2:.4f}")
                print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停
            if patience_counter >= patience:
                print(f"\n早停在 epoch {epoch+1} (连续{patience}个epoch未提升)")
                break
        
        # 加载最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            print(f"\n加载最佳模型 (RMSE: {best_rmse:.4f}, R²: {best_r2:.4f})")
        
        # 最终验证
        _, final_mae, final_rmse, final_r2, val_preds, val_targets = self.validate(
            self.model, val_loader, criterion
        )
        
        print(f"\n最终验证指标:")
        print(f"MAE: {final_mae:.4f}")
        print(f"RMSE: {final_rmse:.4f}")
        print(f"R²分数: {final_r2:.4f}")
        
        # 计算得分
        score = 1.0 / (1.0 + final_rmse)
        print(f"Score = 1.0 / (1.0 + RMSE) = {score:.6f}")
        
        return final_rmse, score, train_losses, val_losses, val_rmses
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        预测测试集
        """
        print("\n预测测试集...")
        
        # 对测试集进行特征工程
        if self.use_advanced_features:
            test_df_processed = self.create_advanced_features(test_df, is_train=False)
        else:
            test_df_processed = test_df.copy()
        
        # 准备测试特征
        X_test = self.prepare_features(test_df_processed, fit_scaler=False, fit_selector=False)
        print(f"测试特征形状: {X_test.shape}")
        
        # 转换为PyTorch张量
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # 预测
        self.model.eval()
        
        with torch.no_grad():
            predictions = self.model(X_test_tensor)
        
        predictions = predictions.cpu().numpy()
        
        # 后处理：确保没有负值（因为阳性率不能为负）
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def create_submission(self, test_df: pd.DataFrame, predictions: np.ndarray, 
                         output_path: str = 'submission.csv'):
        """
        创建提交文件
        """
        submission = pd.DataFrame({
            'id': test_df['id'].values,
            'tested_positive': predictions
        })
        
        submission.to_csv(output_path, index=False)
        print(f"\n提交文件已保存到: {output_path}")
        print(f"提交文件形状: {submission.shape}")
        print(f"预测统计:")
        print(f"  均值: {predictions.mean():.4f}")
        print(f"  标准差: {predictions.std():.4f}")
        print(f"  最小值: {predictions.min():.4f}")
        print(f"  最大值: {predictions.max():.4f}")
        
        return submission

def main():
    """主函数"""
    # 文件路径
    train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
    test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'
    
    # 创建增强版预测器
    predictor = EnhancedCOVID19Predictor(use_advanced_features=True)
    
    # 加载数据
    train_df, test_df = predictor.load_data(train_path, test_path)
    
    # 划分训练集和验证集（时间序列顺序）
    train_data, val_data = predictor.create_dataset(train_df, val_ratio=0.2)
    
    # 训练模型
    rmse, score, train_losses, val_losses, val_rmses = predictor.train(
        train_data, val_data, n_epochs=200, lr=0.001, batch_size=64
    )
    
    # 预测测试集
    predictions = predictor.predict(test_df)
    
    # 创建提交文件
    submission = predictor.create_submission(test_df, predictions, 'submission.csv')
    
    # 打印最终得分
    print(f"\n{'='*60}")
    print(f"最终验证分数: Score = 1.0 / (1.0 + RMSE) = {score:.6f}")
    print(f"验证集RMSE: {rmse:.6f}")
    print(f"{'='*60}")
    
    # 输出分数（必须的最后一行）
    print(f"Score= {score}")

if __name__ == '__main__':
    main()