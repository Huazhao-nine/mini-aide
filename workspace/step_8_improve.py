#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COVID-19 新增病例百分比预测 - Kaggle Grandmaster 终极优化版
目标分数：0.8+
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

warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

class COVID19Predictor:
    def __init__(self, n_features: int = 30, device: str = None):
        """
        初始化预测器 - 简化但更有效
        
        Args:
            n_features: 特征选择的数量
            device: 计算设备
        """
        self.n_features = n_features
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 特征处理组件
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_regression, k=n_features)
        self.selected_features = None
        self.feature_columns = None
        
        # 模型
        self.model = None
        
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
        
        # 目标变量统计
        if 'tested_positive_day3' in train_df.columns:
            target_stats = train_df['tested_positive_day3'].describe()
            print(f"\n目标变量统计:")
            print(f"  均值: {target_stats['mean']:.4f}")
            print(f"  标准差: {target_stats['std']:.4f}")
            print(f"  最小值: {target_stats['min']:.4f}")
            print(f"  最大值: {target_stats['max']:.4f}")
        
        return train_df, test_df
    
    def create_smart_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建智能特征工程 - 基于领域知识
        """
        df = df.copy()
        
        # 1. 基本时间特征
        time_features = ['cli', 'ili', 'hh_cmnty_cli', 'nohh_cmnty_cli', 
                        'tested_positive', 'wearing_mask_7d']
        
        for feat in time_features:
            # 检查是否有三天的数据
            day_cols = [f"{feat}_day{i}" for i in [1, 2, 3] 
                       if f"{feat}_day{i}" in df.columns]
            
            if len(day_cols) >= 2:
                values = df[day_cols].values
                
                # 均值
                df[f"{feat}_mean"] = values.mean(axis=1)
                
                # 趋势 (day3 - day1)
                if len(day_cols) == 3:
                    df[f"{feat}_trend"] = values[:, 2] - values[:, 0]
                
                # 最近值 (day3)
                df[f"{feat}_latest"] = values[:, -1] if len(day_cols) == 3 else values[:, -1]
        
        # 2. 关键交互特征（基于领域知识）
        # 症状与防护行为的交互
        if all(col in df.columns for col in ['cli_day3', 'wearing_mask_7d_day3']):
            df['cli_mask_interaction'] = df['cli_day3'] * (100 - df['wearing_mask_7d_day3']) / 100
        
        # 担心感染与室内活动的交互
        if all(col in df.columns for col in ['wworried_catch_covid_day3', 'wlarge_event_indoors_day3']):
            df['worry_indoor_interaction'] = df['wworried_catch_covid_day3'] * df['wlarge_event_indoors_day3'] / 100
        
        # 疫苗接种朋友比例与症状的交互
        if all(col in df.columns for col in ['wcovid_vaccinated_friends_day3', 'cli_day3']):
            df['vaccine_symptom_interaction'] = (100 - df['wcovid_vaccinated_friends_day3']) * df['cli_day3'] / 100
        
        # 3. 复合风险指标
        # 风险 = 症状 × (1 - 防护)
        if all(col in df.columns for col in ['cli_mean', 'wearing_mask_7d_mean', 
                                            'wbelief_masking_effective_day3']):
            protection_score = (df['wearing_mask_7d_mean'] + df['wbelief_masking_effective_day3']) / 2
            df['risk_score'] = df['cli_mean'] * (100 - protection_score) / 100
        
        # 4. 前两天的目标变量衍生特征（如果存在）
        if 'tested_positive_day1' in df.columns and 'tested_positive_day2' in df.columns:
            df['tested_positive_growth'] = df['tested_positive_day2'] - df['tested_positive_day1']
            df['tested_positive_ratio'] = df['tested_positive_day2'] / (df['tested_positive_day1'] + 1e-10)
            
            # 动量指标
            if 'tested_positive_trend' in df.columns:
                df['tested_positive_momentum'] = df['tested_positive_trend'] / (df['tested_positive_mean'] + 1e-10)
        
        # 5. 行为聚合指标
        behavior_cols = [col for col in df.columns if any(x in col for x in ['indoors', 'transit', 'mask'])]
        if behavior_cols:
            # 计算行为指标的平均值
            df['behavior_avg'] = df[behavior_cols].mean(axis=1)
        
        # 6. 信念聚合指标
        belief_cols = [col for col in df.columns if 'belief' in col or 'worried' in col]
        if belief_cols:
            df['belief_avg'] = df[belief_cols].mean(axis=1)
        
        # 移除原始的时间序列列，保留衍生特征
        # 注意：我们保留所有原始列，让特征选择器决定
        
        print(f"特征工程后特征数量: {df.shape[1]}")
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit_scaler: bool = False, 
                        fit_selector: bool = False, y: np.ndarray = None) -> np.ndarray:
        """
        准备特征 - 简化版
        """
        df = df.copy()
        
        # 移除ID列和目标列
        columns_to_drop = ['id']
        if 'tested_positive_day3' in df.columns:
            columns_to_drop.append('tested_positive_day3')
        
        # 只移除存在的列
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        
        # 保存特征列名
        if fit_scaler:
            self.feature_columns = [col for col in df.columns if col not in columns_to_drop]
        
        # 分离特征
        X = df.drop(columns=columns_to_drop, errors='ignore')
        
        # 在测试阶段，确保列顺序与训练时一致
        if not fit_scaler and hasattr(self, 'feature_columns') and self.feature_columns is not None:
            missing_cols = set(self.feature_columns) - set(X.columns)
            if missing_cols:
                print(f"警告: 测试集缺少 {len(missing_cols)} 个特征，用0填充")
                for col in missing_cols:
                    X[col] = 0
            X = X[self.feature_columns]
        
        # 转换为numpy数组
        features = X.values
        
        # 处理NaN和无限值
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        # 标准化特征
        if fit_scaler:
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)
        
        # 特征选择
        if fit_selector and y is not None:
            features = self.feature_selector.fit_transform(features, y)
            self.selected_features = self.feature_selector.get_support(indices=True)
            print(f"选择了 {len(self.selected_features)} 个最佳特征")
            
            # 输出最重要的特征
            if hasattr(self.feature_selector, 'scores_'):
                feature_scores = pd.DataFrame({
                    'feature': X.columns,
                    'score': self.feature_selector.scores_
                }).sort_values('score', ascending=False)
                print("\nTop 15特征:")
                print(feature_scores.head(15).to_string())
        elif hasattr(self, 'selected_features') and self.selected_features is not None:
            features = features[:, self.selected_features]
        
        return features
    
    def build_model(self, input_dim: int) -> nn.Module:
        """
        构建简单但有效的神经网络模型
        """
        class COVIDNet(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                
                # 更简单的架构，减少过拟合风险
                self.network = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    
                    nn.Linear(64, 32),
                    nn.BatchNorm1d(32),
                    nn.ReLU(),
                    
                    nn.Linear(32, 1)
                )
                
                # 初始化权重
                self._initialize_weights()
            
            def _initialize_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm1d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                return self.network(x).squeeze()
        
        return COVIDNet(input_dim).to(self.device)
    
    def train_model(self, X_train, y_train, X_val, y_val, n_epochs: int = 200):
        """
        训练模型
        """
        print(f"\n开始训练模型...")
        print(f"训练集大小: {X_train.shape[0]}, 验证集大小: {X_val.shape[0]}")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 创建数据集和数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        batch_size = min(32, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                 shuffle=False, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=0)
        
        # 构建模型
        self.model = self.build_model(X_train.shape[1])
        
        # 使用Huber损失（对异常值更鲁棒）
        criterion = nn.HuberLoss(delta=2.0)
        
        # 优化器
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001, 
                               weight_decay=1e-4)
        
        # 学习率调度
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                        patience=10, factor=0.5)
        
        # 训练循环
        best_val_rmse = float('inf')
        best_model_state = None
        patience = 20
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # 训练模式
            self.model.train()
            train_loss = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 验证模式
            self.model.eval()
            val_loss = 0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for data, target in val_loader:
                    output = self.model(data)
                    loss = criterion(output, target)
                    val_loss += loss.item()
                    
                    val_preds.extend(output.cpu().numpy())
                    val_targets.extend(target.cpu().numpy())
            
            val_loss /= len(val_loader)
            
            # 计算RMSE
            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            val_rmse = np.sqrt(np.mean((val_preds - val_targets) ** 2))
            
            # 学习率调度
            scheduler.step(val_rmse)
            
            # 保存最佳模型
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"早停在 epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 20 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, "
                      f"Val RMSE: {val_rmse:.4f}, LR: {current_lr:.6f}")
        
        # 加载最佳模型
        self.model.load_state_dict(best_model_state)
        
        # 计算最终验证得分
        final_score = 1.0 / (1.0 + best_val_rmse)
        print(f"\n最佳验证RMSE: {best_val_rmse:.6f}")
        print(f"最佳验证得分: {final_score:.6f}")
        
        return best_val_rmse, final_score
    
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        预测测试集
        """
        print("\n预测测试集...")
        
        # 对测试集进行特征工程
        test_df_processed = self.create_smart_features(test_df)
        
        # 准备测试特征
        X_test = self.prepare_features(test_df_processed, fit_scaler=False, 
                                      fit_selector=False)
        print(f"测试特征形状: {X_test.shape}")
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            predictions = self.model(X_test_tensor).cpu().numpy()
        
        # 后处理
        predictions = self.post_process_predictions(predictions, test_df)
        
        return predictions
    
    def post_process_predictions(self, predictions: np.ndarray, test_df: pd.DataFrame) -> np.ndarray:
        """
        智能后处理预测结果
        """
        # 1. 确保没有负值
        predictions = np.maximum(predictions, 0)
        
        # 2. 基于前两天的目标值进行平滑
        if 'tested_positive_day1' in test_df.columns and 'tested_positive_day2' in test_df.columns:
            day1 = test_df['tested_positive_day1'].values
            day2 = test_df['tested_positive_day2'].values
            
            # 计算简单趋势（线性外推）
            trend = day2 - day1
            linear_pred = day2 + trend * 0.5  # 假设50%的趋势延续
            
            # 混合预测：70%模型预测 + 30%线性外推
            predictions = 0.7 * predictions + 0.3 * linear_pred
        
        # 3. 确保预测值在合理范围内（基于训练集统计）
        # 训练集目标变量范围：3.0543 到 46.9525
        predictions = np.clip(predictions, 2.0, 50.0)
        
        # 4. 轻微平滑（移动平均）
        window_size = 3
        if len(predictions) > window_size:
            smoothed = np.convolve(predictions, np.ones(window_size)/window_size, mode='same')
            # 保留边缘值不变
            predictions[window_size//2:-window_size//2] = smoothed[window_size//2:-window_size//2]
        
        return predictions
    
    def evaluate_on_validation(self, X_val, y_val):
        """
        在验证集上评估模型
        """
        self.model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            predictions = self.model(X_val_tensor).cpu().numpy()
        
        # 计算指标
        rmse = np.sqrt(np.mean((predictions - y_val) ** 2))
        mae = np.mean(np.abs(predictions - y_val))
        
        print(f"\n验证集评估:")
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        return rmse
    
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
        print(f"\n预测值统计:")
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
    
    # 创建预测器
    predictor = COVID19Predictor(n_features=25)
    
    # 加载数据
    train_df, test_df = predictor.load_data(train_path, test_path)
    
    # 划分训练集和验证集（时间序列顺序）
    split_idx = int(len(train_df) * 0.8)
    train_data = train_df.iloc[:split_idx].copy()
    val_data = train_df.iloc[split_idx:].copy()
    
    print(f"\n训练集大小: {len(train_data)}")
    print(f"验证集大小: {len(val_data)}")
    
    # 特征工程
    print("\n对训练集进行特征工程...")
    train_data_processed = predictor.create_smart_features(train_data)
    val_data_processed = predictor.create_smart_features(val_data)
    
    # 准备训练数据
    print("\n准备特征...")
    y_train = train_data['tested_positive_day3'].values
    X_train = predictor.prepare_features(train_data_processed, fit_scaler=True, 
                                       fit_selector=True, y=y_train)
    
    # 准备验证数据
    y_val = val_data['tested_positive_day3'].values
    X_val = predictor.prepare_features(val_data_processed, fit_scaler=False, 
                                     fit_selector=False)
    
    print(f"训练特征形状: {X_train.shape}")
    print(f"验证特征形状: {X_val.shape}")
    
    # 训练模型
    rmse, score = predictor.train_model(X_train, y_train, X_val, y_val, n_epochs=200)
    
    # 在验证集上评估
    predictor.evaluate_on_validation(X_val, y_val)
    
    # 预测测试集
    predictions = predictor.predict(test_df)
    
    # 创建提交文件
    submission = predictor.create_submission(test_df, predictions, 'submission.csv')
    
    # 打印最终得分
    print(f"\n{'='*60}")
    print(f"最终验证分数: Score = 1.0 / (1.0 + RMSE) = {score:.6f}")
    print(f"验证集RMSE: {rmse:.6f}")
    print(f"{'='*60}")
    
    return score

if __name__ == '__main__':
    final_score = main()
    # 最后一行打印分数
    print(f"Score= (1.0 / (1.0 + RMSE)) = {final_score:.6f}")