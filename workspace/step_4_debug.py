#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COVID-19 新增病例百分比预测
Kaggle Grandmaster 解决方案
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

class COVID19Predictor:
    def __init__(self, n_features: int = 15, device: str = None):
        """
        初始化预测器
        
        Args:
            n_features: 特征选择的数量
            device: 计算设备 (cpu/cuda)
        """
        self.n_features = n_features
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 特征工程组件
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_regression, k=n_features)
        
        # 模型
        self.model = None
        self.selected_features = None
        self.feature_columns = None  # 保存特征列名
        self.interaction_pairs = None  # 保存交互特征对
        self.trend_features = None  # 保存趋势特征
        
    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载数据
        
        Returns:
            train_df: 训练数据
            test_df: 测试数据
            submission_template: 提交模板
        """
        print("正在加载数据...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"训练集形状: {train_df.shape}")
        print(f"测试集形状: {test_df.shape}")
        
        return train_df, test_df
    
    def create_interaction_features(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """
        创建交互特征
        
        Args:
            df: 输入数据
            is_train: 是否为训练阶段（如果是训练阶段，会保存特征定义）
        """
        df = df.copy()
        
        # 定义交互特征对 - 基于训练数据中存在的特征
        if is_train or self.interaction_pairs is None:
            # 只有在训练阶段或未定义时才定义交互特征对
            interaction_pairs = []
            
            # 检查特征是否存在
            base_features = []
            for day in [1, 2, 3]:
                if f'cli_day{day}' in df.columns and f'wearing_mask_7d_day{day}' in df.columns:
                    interaction_pairs.append(('cli', 'wearing_mask_7d'))
                    break
            
            for day in [1, 2, 3]:
                if f'ili_day{day}' in df.columns and f'wearing_mask_7d_day{day}' in df.columns:
                    interaction_pairs.append(('ili', 'wearing_mask_7d'))
                    break
            
            for day in [1, 2, 3]:
                if f'cli_day{day}' in df.columns and f'wbelief_masking_effective_day{day}' in df.columns:
                    interaction_pairs.append(('cli', 'wbelief_masking_effective'))
                    break
            
            for day in [1, 2, 3]:
                if f'ili_day{day}' in df.columns and f'wbelief_masking_effective_day{day}' in df.columns:
                    interaction_pairs.append(('ili', 'wbelief_masking_effective'))
                    break
            
            for day in [1, 2, 3]:
                if f'cli_day{day}' in df.columns and f'wbelief_distancing_effective_day{day}' in df.columns:
                    interaction_pairs.append(('cli', 'wbelief_distancing_effective'))
                    break
            
            for day in [1, 2, 3]:
                if f'hh_cmnty_cli_day{day}' in df.columns and f'wbelief_masking_effective_day{day}' in df.columns:
                    interaction_pairs.append(('hh_cmnty_cli', 'wbelief_masking_effective'))
                    break
            
            for day in [1, 2, 3]:
                if f'wearing_mask_7d_day{day}' in df.columns and f'wbelief_masking_effective_day{day}' in df.columns:
                    interaction_pairs.append(('wearing_mask_7d', 'wbelief_masking_effective'))
                    break
            
            # 去重
            interaction_pairs = list(set(interaction_pairs))
            
            if is_train:
                self.interaction_pairs = interaction_pairs
        else:
            # 使用训练阶段保存的交互特征对
            interaction_pairs = self.interaction_pairs
        
        # 为每一天创建交互特征
        for day in [1, 2, 3]:
            for feat1, feat2 in interaction_pairs:
                col1 = f"{feat1}_day{day}"
                col2 = f"{feat2}_day{day}"
                
                # 检查列是否存在
                if col1 in df.columns and col2 in df.columns:
                    interaction_name = f"{feat1}_{feat2}_day{day}"
                    df[interaction_name] = df[col1] * df[col2]
        
        # 创建跨天的特征（趋势特征）
        if is_train or self.trend_features is None:
            # 定义趋势特征
            trend_features = []
            for feat in ['cli', 'ili', 'wearing_mask_7d', 
                        'wbelief_masking_effective', 'wworried_catch_covid']:
                if f"{feat}_day1" in df.columns and f"{feat}_day2" in df.columns and f"{feat}_day3" in df.columns:
                    trend_features.append(feat)
            
            # 只在训练集中包含tested_positive
            if 'tested_positive_day1' in df.columns and 'tested_positive_day2' in df.columns and 'tested_positive_day3' in df.columns:
                trend_features.append('tested_positive')
            
            if is_train:
                self.trend_features = trend_features
        else:
            trend_features = self.trend_features
        
        for feat in trend_features:
            col1 = f"{feat}_day1"
            col2 = f"{feat}_day2"
            col3 = f"{feat}_day3"
            
            # 确保所有需要的列都存在
            if col1 in df.columns and col2 in df.columns and col3 in df.columns:
                # 趋势（差分）
                df[f"{feat}_trend_1_2"] = df[col2] - df[col1]
                df[f"{feat}_trend_2_3"] = df[col3] - df[col2]
                df[f"{feat}_trend_1_3"] = df[col3] - df[col1]
                
                # 移动平均
                df[f"{feat}_avg"] = (df[col1] + df[col2] + df[col3]) / 3
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False, 
                         fit_selector: bool = False, y: np.ndarray = None) -> np.ndarray:
        """
        准备特征
        
        Args:
            df: 输入数据
            fit_scaler: 是否拟合scaler
            fit_selector: 是否拟合特征选择器
            y: 目标变量（用于特征选择）
            
        Returns:
            features: 特征矩阵
        """
        df = df.copy()
        
        # 移除ID列和目标列
        columns_to_drop = ['id']
        if 'tested_positive_day3' in df.columns:
            columns_to_drop.append('tested_positive_day3')
        
        # 记录原始特征列
        original_columns = [col for col in df.columns if col not in columns_to_drop]
        
        # 只移除存在的列
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=columns_to_drop)
        
        # 保存特征列名（只在训练阶段）
        if fit_scaler:
            self.feature_columns = df.columns.tolist()
        
        # 在测试阶段，确保列顺序与训练时一致
        if not fit_scaler and hasattr(self, 'feature_columns') and self.feature_columns is not None:
            # 检查是否有缺失的列
            missing_cols = set(self.feature_columns) - set(df.columns)
            if missing_cols:
                # 添加缺失的列（用0填充）
                for col in missing_cols:
                    df[col] = 0
            # 确保列顺序一致
            df = df[self.feature_columns]
        
        # 转换为numpy数组
        features = df.values
        
        # 标准化特征
        if fit_scaler:
            features = self.scaler.fit_transform(features)
        else:
            # 确保scaler已经拟合
            if hasattr(self.scaler, 'mean_'):
                features = self.scaler.transform(features)
            else:
                raise ValueError("Scaler has not been fitted yet.")
        
        # 特征选择
        if fit_selector and y is not None:
            features = self.feature_selector.fit_transform(features, y)
            self.selected_features = self.feature_selector.get_support(indices=True)
        elif hasattr(self, 'selected_features') and self.selected_features is not None:
            # 使用训练好的特征选择器
            features = features[:, self.selected_features]
        
        return features
    
    def create_dataset(self, train_df: pd.DataFrame, val_ratio: float = 0.2) -> Tuple:
        """
        创建训练和验证数据集（时间序列顺序）
        """
        # 先进行特征工程（训练阶段）
        train_df_processed = self.create_interaction_features(train_df, is_train=True)
        
        # 计算分割点
        split_idx = int(len(train_df_processed) * (1 - val_ratio))
        
        # 分割训练集和验证集（不shuffle！）
        train_data = train_df_processed.iloc[:split_idx].copy()
        val_data = train_df_processed.iloc[split_idx:].copy()
        
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")
        
        return train_data, val_data
    
    def build_model(self, input_dim: int) -> nn.Module:
        """
        构建神经网络模型
        """
        class COVIDNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                
                self.model = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.15),
                    
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    
                    nn.Linear(32, 16),
                    nn.BatchNorm1d(16),
                    nn.ReLU(),
                    
                    nn.Linear(16, 1)
                )
            
            def forward(self, x):
                return self.model(x).squeeze()
        
        return COVIDNet(input_dim).to(self.device)
    
    def train_epoch(self, model, train_loader, criterion, optimizer, epoch):
        """训练一个epoch"""
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
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
        
        # 计算RMSE
        mse = np.mean((np.array(all_preds) - np.array(all_targets)) ** 2)
        rmse = np.sqrt(mse)
        
        return total_loss / len(val_loader), rmse, all_preds, all_targets
    
    def train(self, train_data, val_data, n_epochs: int = 200, 
              lr: float = 0.001, batch_size: int = 32):
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
                                 shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0)
        
        # 构建模型
        self.model = self.build_model(X_train.shape[1])
        
        # 定义损失函数和优化器
        criterion = nn.L1Loss()  # MAE损失，更抗噪
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        factor=0.5, patience=10, 
                                                        verbose=True)
        
        print("\n开始训练...")
        best_rmse = float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # 训练
            train_loss = self.train_epoch(self.model, train_loader, criterion, optimizer, epoch)
            
            # 验证
            val_loss, val_rmse, _, _ = self.validate(self.model, val_loader, criterion)
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"\n早停在 epoch {epoch+1}")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, "
                      f"Val RMSE: {val_rmse:.4f}")
        
        # 加载最佳模型
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # 最终验证
        _, final_rmse, val_preds, val_targets = self.validate(self.model, val_loader, criterion)
        
        print(f"\n最佳验证RMSE: {final_rmse:.4f}")
        
        # 计算得分
        score = 1.0 / (1.0 + final_rmse)
        print(f"Score = 1.0 / (1.0 + RMSE) = {score:.6f}")
        
        return final_rmse, score
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        预测测试集
        """
        print("\n预测测试集...")
        
        # 对测试集进行特征工程（使用训练阶段保存的特征定义）
        test_df_processed = self.create_interaction_features(test_df, is_train=False)
        
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
        
        return submission

def main():
    """主函数"""
    # 文件路径
    train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
    test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'
    
    # 创建预测器
    predictor = COVID19Predictor(n_features=15)
    
    # 加载数据
    train_df, test_df = predictor.load_data(train_path, test_path)
    
    # 划分训练集和验证集（时间序列顺序）
    train_data, val_data = predictor.create_dataset(train_df, val_ratio=0.2)
    
    # 训练模型
    rmse, score = predictor.train(train_data, val_data, n_epochs=200, 
                                  lr=0.001, batch_size=32)
    
    # 预测测试集
    predictions = predictor.predict(test_df)
    
    # 创建提交文件
    submission = predictor.create_submission(test_df, predictions, 'submission.csv')
    
    # 打印最终得分
    print(f"\n{'='*50}")
    print(f"最终验证分数: Score = 1.0 / (1.0 + RMSE) = {score:.6f}")
    print(f"验证集RMSE: {rmse:.6f}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
    # 最后一行打印分数（由训练过程中的输出提供）