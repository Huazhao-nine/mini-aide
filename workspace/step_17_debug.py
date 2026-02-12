#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COVID-19 新增病例百分比预测 - Kaggle Grandmaster 修复版
目标分数：0.45+
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
import torch.nn.functional as F
from sklearn.model_selection import KFold
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

class OptimizedCOVID19Predictor:
    def __init__(self, device: str = None):
        """
        初始化预测器
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 特征处理组件
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns = None
        self.models = []
        self.feature_selector = SelectKBest(f_regression, k=15)  # 按照要求选择15个特征
        
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
        
        return train_df, test_df
    
    def create_smart_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        创建智能特征工程 - 根据要求构建交互特征
        """
        df = df.copy()
        
        # 1. 首先识别基础特征名称
        # 从列名中提取基础特征（移除_dayX后缀）
        all_columns = df.columns.tolist()
        
        # 找出所有带_day后缀的列
        day_columns = [col for col in all_columns if '_day' in col]
        
        # 提取基础特征名称（移除_day1, _day2, _day3后缀）
        base_features = set()
        for col in day_columns:
            if '_day1' in col:
                base_feature = col.replace('_day1', '')
                base_features.add(base_feature)
            elif '_day2' in col:
                base_feature = col.replace('_day2', '')
                base_features.add(base_feature)
            elif '_day3' in col:
                base_feature = col.replace('_day3', '')
                base_features.add(base_feature)
        
        base_features = list(base_features)
        
        # 2. 为每个基础特征计算统计量
        for base_feature in base_features:
            day_cols = []
            for day in [1, 2, 3]:
                col_name = f"{base_feature}_day{day}"
                if col_name in df.columns:
                    day_cols.append(col_name)
            
            if len(day_cols) >= 2:
                # 计算平均值
                df[f'{base_feature}_mean'] = df[day_cols].mean(axis=1)
                # 计算标准差
                df[f'{base_feature}_std'] = df[day_cols].std(axis=1)
                # 计算趋势（最后一天 - 第一天）
                if len(day_cols) == 3:
                    df[f'{base_feature}_trend'] = df[day_cols[2]] - df[day_cols[0]]
        
        # 3. 构建交互特征（按照要求）
        # cli * wearing_mask_7d 交互特征
        if 'cli_day1' in df.columns and 'wearing_mask_7d_day1' in df.columns:
            for day in [1, 2, 3]:
                cli_col = f'cli_day{day}'
                mask_col = f'wearing_mask_7d_day{day}'
                if cli_col in df.columns and mask_col in df.columns:
                    df[f'cli_mask_interaction_day{day}'] = df[cli_col] * df[mask_col]
            
            # 计算交互特征的平均值
            interaction_cols = [f'cli_mask_interaction_day{day}' for day in [1, 2, 3] 
                              if f'cli_mask_interaction_day{day}' in df.columns]
            if interaction_cols:
                df['cli_mask_interaction_mean'] = df[interaction_cols].mean(axis=1)
        
        # 其他可能的交互特征
        if 'wbelief_masking_effective_day1' in df.columns and 'wearing_mask_7d_day1' in df.columns:
            for day in [1, 2, 3]:
                belief_col = f'wbelief_masking_effective_day{day}'
                mask_col = f'wearing_mask_7d_day{day}'
                if belief_col in df.columns and mask_col in df.columns:
                    df[f'belief_mask_interaction_day{day}'] = df[belief_col] * df[mask_col]
        
        # 4. 如果是训练集，保留目标列
        if is_train and 'tested_positive_day3' in df.columns:
            # 保留目标列
            pass
        
        # 5. 移除原始的时间序列特征，但保留州特征和衍生特征
        # 先找出所有州特征（两个大写字母的列）
        state_cols = [col for col in df.columns if col.isupper() and len(col) == 2]
        
        # 要保留的列：州特征、衍生特征、目标列（如果是训练集）
        keep_cols = state_cols + ['id']
        
        # 添加衍生特征
        derived_cols = [col for col in df.columns if '_mean' in col or '_std' in col 
                       or '_trend' in col or '_interaction' in col]
        keep_cols.extend(derived_cols)
        
        # 添加目标列（如果是训练集）
        if is_train and 'tested_positive_day3' in df.columns:
            keep_cols.append('tested_positive_day3')
        
        # 删除不在keep_cols中的_day列
        day_cols_to_drop = [col for col in df.columns if '_day' in col and col not in keep_cols]
        df = df.drop(columns=day_cols_to_drop)
        
        print(f"特征工程后特征数量: {df.shape[1]}")
        return df
    
    def select_features(self, X: np.ndarray, y: np.ndarray = None, fit: bool = False) -> np.ndarray:
        """
        选择最重要的特征
        """
        if fit and y is not None:
            X_selected = self.feature_selector.fit_transform(X, y)
            print(f"特征选择: {X.shape[1]} -> {X_selected.shape[1]}")
        else:
            X_selected = self.feature_selector.transform(X)
        
        return X_selected
    
    def prepare_data(self, df: pd.DataFrame, fit: bool = False, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        准备数据
        """
        df = df.copy()
        
        # 分离目标变量
        target_col = 'tested_positive_day3'
        if target_col in df.columns:
            if fit:
                # 对目标变量进行缩放（但不要进行对数变换，以免影响RMSE计算）
                y = df[target_col].values
                y = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                y = df[target_col].values
        else:
            y = None
        
        # 移除ID和目标列
        columns_to_drop = ['id']
        if target_col in df.columns:
            columns_to_drop.append(target_col)
        
        X = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # 保存特征列名
        if fit:
            self.feature_columns = X.columns.tolist()
        
        # 确保列顺序一致
        if not fit and hasattr(self, 'feature_columns'):
            # 添加缺失的列
            missing_cols = set(self.feature_columns) - set(X.columns)
            for col in missing_cols:
                X[col] = 0
            # 重新排列列顺序
            X = X[self.feature_columns]
        
        # 转换为numpy并处理缺失值
        X = X.values
        X = np.nan_to_num(X, nan=0.0)
        
        # 标准化特征
        if fit:
            X = self.scaler.fit_transform(X)
        else:
            X = self.scaler.transform(X)
        
        # 特征选择
        if fit and y is not None:
            X = self.select_features(X, y, fit=True)
        else:
            X = self.select_features(X, fit=False)
        
        return X, y
    
    def build_model(self, input_dim: int) -> nn.Module:
        """
        构建神经网络模型 - 按照要求的简单结构
        """
        class COVIDNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                
                # 按照要求的简单结构：Input -> 64 -> 32 -> 1
                self.model = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    
                    nn.Linear(32, 1)
                )
                
                # 初始化权重
                self._initialize_weights()
            
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm1d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                return self.model(x).squeeze()
        
        return COVIDNet(input_dim).to(self.device)
    
    def train_model(self, X_train, y_train, X_val, y_val, model_idx: int):
        """
        训练单个模型
        """
        print(f"\n训练模型 {model_idx + 1}...")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # 数据加载器 - 注意：时间序列不shuffle！
        batch_size = min(64, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=0)  # shuffle=False
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0)
        
        # 构建模型
        model = self.build_model(X_train.shape[1])
        
        # 损失函数 - 按照要求使用L1Loss (MAE)
        criterion = nn.L1Loss()
        
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=False
        )
        
        # 训练循环
        best_val_rmse = float('inf')
        best_model_state = None
        patience = 30
        patience_counter = 0
        n_epochs = 200
        
        for epoch in range(n_epochs):
            # 训练
            model.train()
            train_loss = 0
            for data, target in train_loader:
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证
            model.eval()
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = model(data)
                    val_preds.extend(output.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())
            
            # 计算RMSE（在原始尺度上）
            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            
            # 逆变换回原始尺度
            val_preds_original = self.target_scaler.inverse_transform(val_preds.reshape(-1, 1)).flatten()
            val_targets_original = self.target_scaler.inverse_transform(val_targets.reshape(-1, 1)).flatten()
            
            # 计算RMSE
            val_rmse = np.sqrt(np.mean((val_preds_original - val_targets_original) ** 2))
            
            # 更新学习率
            scheduler.step(val_rmse)
            
            # 保存最佳模型
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                if epoch > 50:  # 确保至少训练50轮
                    print(f"模型 {model_idx + 1} 早停在 epoch {epoch + 1}")
                    break
            
            # 打印进度
            if (epoch + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                      f"Val RMSE: {val_rmse:.4f}, LR: {current_lr:.6f}")
        
        # 加载最佳模型
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # 计算最终得分
        final_score = 1.0 / (1.0 + best_val_rmse)
        print(f"模型 {model_idx + 1} 最佳验证RMSE: {best_val_rmse:.6f}")
        print(f"模型 {model_idx + 1} 最佳验证得分: {final_score:.6f}")
        
        return model, best_val_rmse
    
    def train_with_kfold(self, X, y, n_folds: int = 5):
        """
        使用K-Fold交叉验证训练模型
        """
        print(f"\n开始K-Fold交叉验证训练 ({n_folds} folds)...")
        
        kf = KFold(n_splits=n_folds, shuffle=False)  # 时间序列不shuffle
        all_models = []
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n{'='*50}")
            print(f"训练 Fold {fold_idx + 1}/{n_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 设置不同的随机种子
            model_seed = SEED + fold_idx * 100
            torch.manual_seed(model_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(model_seed)
            
            model, rmse = self.train_model(X_train, y_train, X_val, y_val, fold_idx)
            all_models.append(model)
            fold_scores.append(rmse)
            
            fold_score = 1.0 / (1.0 + rmse)
            print(f"Fold {fold_idx + 1} RMSE: {rmse:.6f}")
            print(f"Fold {fold_idx + 1} 得分: {fold_score:.6f}")
        
        self.models = all_models
        
        # 计算总体平均分数
        avg_rmse = np.mean(fold_scores)
        avg_score = 1.0 / (1.0 + avg_rmse)
        
        print(f"\n{'='*50}")
        print(f"K-Fold训练完成")
        print(f"平均RMSE: {avg_rmse:.6f}")
        print(f"平均得分: {avg_score:.6f}")
        print(f"总模型数: {len(self.models)}")
        
        return avg_rmse, avg_score
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        使用模型集合进行预测
        """
        if not self.models:
            raise ValueError("没有训练好的模型")
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        all_predictions = []
        
        for i, model in enumerate(self.models):
            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor).cpu().numpy()
                # 逆变换回原始尺度
                predictions_original = self.target_scaler.inverse_transform(
                    predictions.reshape(-1, 1)).flatten()
                all_predictions.append(predictions_original)
        
        # 平均所有模型的预测
        all_predictions = np.array(all_predictions)
        ensemble_predictions = np.mean(all_predictions, axis=0)
        
        print(f"\n集合预测统计:")
        print(f"  预测均值: {ensemble_predictions.mean():.4f}")
        print(f"  预测标准差: {ensemble_predictions.std():.4f}")
        print(f"  模型数量: {len(self.models)}")
        
        return ensemble_predictions
    
    def intelligent_post_process(self, predictions: np.ndarray, test_df: pd.DataFrame) -> np.ndarray:
        """
        智能后处理
        """
        predictions = predictions.copy()
        
        # 1. 确保没有负值（但按照要求不设置上限）
        predictions = np.maximum(predictions, 0.0)
        
        # 2. 如果有历史数据，可以进行一些调整
        if 'tested_positive_day1' in test_df.columns and 'tested_positive_day2' in test_df.columns:
            day1 = test_df['tested_positive_day1'].values
            day2 = test_df['tested_positive_day2'].values
            
            # 计算历史趋势
            historical_trend = day2 - day1
            
            # 轻微调整预测值，考虑历史趋势
            predictions = predictions + historical_trend * 0.1
        
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
        
        # 输出预测统计
        print(f"\n最终预测值统计:")
        stats = {
            '均值': predictions.mean(),
            '标准差': predictions.std(),
            '最小值': predictions.min(),
            '中位数': np.median(predictions),
            '最大值': predictions.max()
        }
        
        for key, value in stats.items():
            print(f"  {key}: {value:.4f}")
        
        return submission

def main():
    """主函数"""
    # 文件路径
    train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
    test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'
    
    # 创建预测器
    predictor = OptimizedCOVID19Predictor()
    
    # 加载数据
    train_df, test_df = predictor.load_data(train_path, test_path)
    
    # 划分训练集和验证集（时间序列顺序，不shuffle）
    split_idx = int(len(train_df) * 0.8)
    train_data = train_df.iloc[:split_idx].copy()
    val_data = train_df.iloc[split_idx:].copy()
    
    print(f"\n训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    
    # 特征工程
    print("\n对训练集进行特征工程...")
    train_data_processed = predictor.create_smart_features(train_data, is_train=True)
    val_data_processed = predictor.create_smart_features(val_data, is_train=True)
    
    print("\n对测试集进行特征工程...")
    test_data_processed = predictor.create_smart_features(test_df, is_train=False)
    
    # 准备数据
    print("\n准备训练数据...")
    X_train, y_train = predictor.prepare_data(train_data_processed, fit=True)
    
    print("\n准备验证数据...")
    X_val, y_val = predictor.prepare_data(val_data_processed, fit=False)
    
    print("\n准备测试数据...")
    X_test, _ = predictor.prepare_data(test_data_processed, fit=False)
    
    print(f"\n数据形状:")
    print(f"训练特征: {X_train.shape}, 训练目标: {y_train.shape if y_train is not None else 'N/A'}")
    print(f"验证特征: {X_val.shape}, 验证目标: {y_val.shape if y_val is not None else 'N/A'}")
    print(f"测试特征: {X_test.shape}")
    
    # 合并训练和验证数据进行K-Fold训练
    X_full = np.vstack([X_train, X_val])
    y_full = np.concatenate([y_train, y_val])
    
    # 使用K-Fold训练
    print("\n开始训练模型...")
    kfold_rmse, kfold_score = predictor.train_with_kfold(X_full, y_full, n_folds=3)
    
    # 在验证集上评估最终模型
    print("\n在验证集上评估最终模型...")
    val_predictions = predictor.predict_ensemble(X_val)
    
    # 将预测值转换回原始尺度
    val_predictions_original = predictor.target_scaler.inverse_transform(
        val_predictions.reshape(-1, 1)).flatten()
    
    # 计算验证集RMSE
    val_targets_original = predictor.target_scaler.inverse_transform(
        y_val.reshape(-1, 1)).flatten()
    final_val_rmse = np.sqrt(np.mean((val_predictions_original - val_targets_original) ** 2))
    final_val_score = 1.0 / (1.0 + final_val_rmse)
    
    print(f"最终验证集RMSE: {final_val_rmse:.6f}")
    print(f"最终验证集得分: {final_val_score:.6f}")
    
    # 预测测试集
    print("\n预测测试集...")
    predictions = predictor.predict_ensemble(X_test)
    
    # 智能后处理
    predictions = predictor.intelligent_post_process(predictions, test_df)
    
    # 创建提交文件
    submission = predictor.create_submission(test_df, predictions, 'submission.csv')
    
    # 打印最终得分
    print(f"\n{'='*60}")
    print(f"K-Fold平均分数: Score = 1.0 / (1.0 + RMSE) = {kfold_score:.6f}")
    print(f"最终验证集分数: Score = 1.0 / (1.0 + RMSE) = {final_val_score:.6f}")
    print(f"{'='*60}")
    
    return final_val_score

if __name__ == '__main__':
    final_score = main()
    # 最后一行打印分数
    print(f"Score= (1.0 / (1.0 + RMSE)) = {final_score:.6f}")