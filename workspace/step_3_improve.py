import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

# 设置随机种子保证可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# 数据加载和预处理
class COVIDDataset(Dataset):
    def __init__(self, features, target=None):
        self.features = torch.FloatTensor(features)
        self.target = torch.FloatTensor(target) if target is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.target is not None:
            return self.features[idx], self.target[idx]
        return self.features[idx]

# 改进的神经网络模型 - 更深的架构
class COVIDModel(nn.Module):
    def __init__(self, input_size):
        super(COVIDModel, self).__init__()
        
        # 使用残差连接
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.output_layer = nn.Linear(32, 1)
        
        # 残差连接
        self.skip1 = nn.Linear(input_size, 128) if input_size != 128 else nn.Identity()
        self.skip2 = nn.Linear(128, 64) if 128 != 64 else nn.Identity()
        
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        
        # 残差连接
        if x.shape[1] != 128:
            residual = self.skip1(x)
        else:
            residual = x
        x2 = x2 + residual
        
        x3 = self.layer3(x2)
        
        # 残差连接
        if x2.shape[1] != 64:
            residual2 = self.skip2(x2)
        else:
            residual2 = x2
        x3 = x3 + residual2
        
        x4 = self.layer4(x3)
        output = self.output_layer(x4)
        
        return output

# 更丰富的特征工程函数
def feature_engineering(df, is_train=True):
    """创建更丰富的交互特征和统计特征"""
    df = df.copy()
    
    # 原始特征列表
    state_cols = [col for col in df.columns if len(col) == 2]
    base_features = []
    
    # 提取所有基础特征（不包括州、id和目标列）
    for col in df.columns:
        if col not in state_cols and col != 'id' and (is_train or 'tested_positive_day3' not in col):
            # 去除_day1, _day2, _day3后缀获取基础特征名
            if col.endswith('_day1') or col.endswith('_day2') or col.endswith('_day3'):
                base_feature = '_'.join(col.split('_')[:-1])
                if base_feature not in base_features:
                    base_features.append(base_feature)
    
    # 为每个基础特征创建跨天统计特征
    for base_feature in base_features:
        day_cols = [f'{base_feature}_day1', f'{base_feature}_day2', f'{base_feature}_day3']
        if all(col in df.columns for col in day_cols):
            # 计算均值
            df[f'{base_feature}_mean'] = df[day_cols].mean(axis=1)
            # 计算标准差
            df[f'{base_feature}_std'] = df[day_cols].std(axis=1)
            # 计算变化趋势（线性回归斜率近似）
            df[f'{base_feature}_trend'] = (df[f'{base_feature}_day3'] - df[f'{base_feature}_day1']) / 2
            # 计算最新变化
            df[f'{base_feature}_last_change'] = df[f'{base_feature}_day3'] - df[f'{base_feature}_day2']
    
    # 创建有意义的交互特征
    # 1. 症状与防护行为的交互
    for day in ['day1', 'day2', 'day3']:
        # 症状与口罩佩戴
        if f'cli_{day}' in df.columns and f'wearing_mask_7d_{day}' in df.columns:
            df[f'cli_mask_interaction_{day}'] = df[f'cli_{day}'] * df[f'wearing_mask_7d_{day}']
        
        # 症状与疫苗接种朋友比例
        if f'cli_{day}' in df.columns and f'wcovid_vaccinated_friends_{day}' in df.columns:
            df[f'cli_vax_interaction_{day}'] = df[f'cli_{day}'] * df[f'wcovid_vaccinated_friends_{day}']
        
        # 担心感染与室内活动
        if f'wworried_catch_covid_{day}' in df.columns and f'restaurant_indoors_{day}' in df.columns:
            df[f'worried_indoors_interaction_{day}'] = df[f'wworried_catch_covid_{day}'] * df[f'restaurant_indoors_{day}']
    
    # 2. 跨天特征交互（例如：症状趋势与口罩趋势）
    for feat1, feat2 in [('cli', 'wearing_mask_7d'), ('ili', 'public_transit'), ('tested_positive', 'wworried_catch_covid')]:
        if f'{feat1}_trend' in df.columns and f'{feat2}_trend' in df.columns:
            df[f'{feat1}_{feat2}_trend_interaction'] = df[f'{feat1}_trend'] * df[f'{feat2}_trend']
    
    # 3. 创建复合特征
    # 风险指数 = 症状 * (1 - 防护行为)
    for day in ['day1', 'day2', 'day3']:
        if f'cli_{day}' in df.columns and f'wearing_mask_7d_{day}' in df.columns:
            df[f'risk_index_{day}'] = df[f'cli_{day}'] * (100 - df[f'wearing_mask_7d_{day}']) / 100
    
    # 添加平方特征（对于重要特征）
    important_features = ['cli', 'ili', 'tested_positive', 'wearing_mask_7d']
    for feat in important_features:
        for day in ['day1', 'day2', 'day3']:
            col = f'{feat}_{day}'
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
    
    # 创建滞后特征（前几天的目标变量）
    if is_train and 'tested_positive_day1' in df.columns and 'tested_positive_day2' in df.columns:
        df['tested_positive_lag1'] = df['tested_positive_day1']
        df['tested_positive_lag2'] = df['tested_positive_day2']
        # 移动平均
        df['tested_positive_ma2'] = (df['tested_positive_day1'] + df['tested_positive_day2']) / 2
    
    return df

# 主函数
def main():
    # 加载数据
    train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
    test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 保存ID列用于提交
    test_ids = test_df['id'].copy()
    
    # 特征工程
    print("进行特征工程...")
    train_df = feature_engineering(train_df, is_train=True)
    test_df = feature_engineering(test_df, is_train=False)
    
    # 分离特征和目标
    target_col = 'tested_positive_day3'
    y_train = train_df[target_col].values
    
    # 移除不需要的列
    X_train = train_df.drop(['id', target_col], axis=1, errors='ignore')
    X_test = test_df.drop(['id'], axis=1, errors='ignore')
    
    # 确保训练集和测试集列一致
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    
    extra_cols = set(X_test.columns) - set(X_train.columns)
    for col in extra_cols:
        X_train[col] = 0
    
    X_test = X_test[X_train.columns]
    
    print(f"特征数量: {X_train.shape[1]}")
    
    # 时间序列分割：前80%训练，后20%验证（禁止shuffle）
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train.iloc[:split_idx].copy()
    y_train_split = y_train[:split_idx]
    X_val = X_train.iloc[split_idx:].copy()
    y_val = y_train[split_idx:]
    
    print(f"训练集大小: {len(X_train_split)}, 验证集大小: {len(X_val)}")
    
    # 标准化特征 - 使用RobustScaler对异常值更鲁棒
    print("标准化特征...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # 特征选择 - 使用更复杂的方法
    print("选择重要特征...")
    
    # 方法1: 使用LassoCV选择特征
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train_split)
    
    # 选择非零系数的特征
    selected_features_lasso = np.abs(lasso.coef_) > 0.001
    X_train_selected = X_train_scaled[:, selected_features_lasso]
    X_val_selected = X_val_scaled[:, selected_features_lasso]
    X_test_selected = X_test_scaled[:, selected_features_lasso]
    
    print(f"Lasso选择了 {np.sum(selected_features_lasso)} 个特征")
    
    # 方法2: 如果Lasso选择特征太少，使用PCA降维
    if np.sum(selected_features_lasso) < 20:
        print("Lasso选择的特征太少，使用PCA...")
        n_components = min(50, X_train_scaled.shape[1])
        pca = PCA(n_components=n_components)
        X_train_selected = pca.fit_transform(X_train_scaled)
        X_val_selected = pca.transform(X_val_scaled)
        X_test_selected = pca.transform(X_test_scaled)
        print(f"PCA保留了 {n_components} 个主成分，解释方差: {np.sum(pca.explained_variance_ratio_):.2%}")
    
    print(f"最终特征维度: {X_train_selected.shape[1]}")
    
    # 创建数据加载器
    train_dataset = COVIDDataset(X_train_selected, y_train_split)
    val_dataset = COVIDDataset(X_val_selected, y_val)
    test_dataset = COVIDDataset(X_test_selected)
    
    # 设置数据加载器（shuffle=False用于时间序列）
    batch_size = 64  # 增加批次大小
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    input_size = X_train_selected.shape[1]
    model = COVIDModel(input_size).to(device)
    
    # 损失函数 - 使用Huber损失，对异常值更鲁棒
    criterion = nn.HuberLoss(delta=1.0)
    # 同时计算MSE用于监控
    mse_criterion = nn.MSELoss()
    
    # 优化器 - 使用AdamW，更好的权重衰减
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器 - 使用余弦退火
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    
    # 训练参数
    n_epochs = 300
    best_val_loss = float('inf')
    best_val_rmse = float('inf')
    patience = 30
    patience_counter = 0
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    val_rmses = []
    
    print("开始训练模型...")
    
    for epoch in range(n_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_mse = 0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Train]', leave=False)
        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X).squeeze()
            loss = criterion(predictions, batch_y)
            mse_loss = mse_criterion(predictions, batch_y)
            
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_mse += mse_loss.item()
            train_batches += 1
            
            progress_bar.set_postfix({
                'loss': train_loss / train_batches,
                'mse': train_mse / train_batches
            })
        
        # 更新学习率
        scheduler.step()
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_mse = 0
        val_batches = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f'Epoch {epoch+1}/{n_epochs} [Val]', leave=False)
            for batch_X, batch_y in progress_bar_val:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X).squeeze()
                
                loss = criterion(predictions, batch_y)
                mse_loss = mse_criterion(predictions, batch_y)
                
                val_loss += loss.item()
                val_mse += mse_loss.item()
                val_batches += 1
                
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                
                progress_bar_val.set_postfix({
                    'loss': val_loss / val_batches,
                    'mse': val_mse / val_batches
                })
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        val_rmse = np.sqrt(val_mse / val_batches)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_rmses.append(val_rmse)
        
        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Val Loss: {avg_val_loss:.4f}')
            print(f'  Val RMSE: {val_rmse:.4f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 早停策略
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            
            # 保存最佳模型
            torch.save(best_model_state, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'早停在第{epoch+1}轮')
                print(f'最佳验证RMSE: {best_val_rmse:.4f}')
                break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 最终验证集评估
    model.eval()
    final_val_predictions = []
    final_val_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X).squeeze()
            final_val_predictions.extend(predictions.cpu().numpy())
            final_val_targets.extend(batch_y.cpu().numpy())
    
    # 计算最终RMSE和分数
    final_val_rmse = np.sqrt(mean_squared_error(final_val_targets, final_val_predictions))
    score = 1.0 / (1.0 + final_val_rmse)
    
    print(f"\n最终验证集评估:")
    print(f"RMSE: {final_val_rmse:.4f}")
    print(f"MAE: {np.mean(np.abs(np.array(final_val_targets) - np.array(final_val_predictions))):.4f}")
    
    # 生成测试集预测
    model.eval()
    test_predictions = []
    
    with torch.no_grad():
        progress_bar_test = tqdm(test_loader, desc='生成测试集预测')
        for batch_X in progress_bar_test:
            batch_X = batch_X.to(device)
            predictions = model(batch_X).squeeze()
            test_predictions.extend(predictions.cpu().numpy())
    
    # 创建提交文件
    submission = pd.DataFrame({
        'id': test_ids,
        'tested_positive': test_predictions
    })
    
    # 后处理：确保预测值为非负（根据约束，不设置上限）
    submission['tested_positive'] = submission['tested_positive'].clip(lower=0)
    
    # 应用时间序列平滑（对同一州的预测）
    print("应用时间序列平滑...")
    
    # 识别州列
    state_cols = [col for col in train_df.columns if len(col) == 2]
    
    # 为测试集添加州信息
    state_data = test_df[state_cols]
    
    # 对每个州的预测进行简单平滑
    for state in state_cols:
        state_mask = state_data[state] == 1
        if state_mask.sum() > 1:  # 该州有多个样本
            state_indices = submission[state_mask].index
            if len(state_indices) > 2:
                # 对预测值进行移动平均平滑
                predictions = submission.loc[state_indices, 'tested_positive'].values
                smoothed = np.convolve(predictions, np.ones(3)/3, mode='same')
                # 保持第一个和最后一个值不变
                smoothed[0] = predictions[0]
                smoothed[-1] = predictions[-1]
                submission.loc[state_indices, 'tested_positive'] = smoothed
    
    submission.to_csv('submission.csv', index=False)
    
    # 输出最终分数
    print(f"\nScore= (1.0 / (1.0 + RMSE)) = {score:.4f}")
    print(f"验证集RMSE: {final_val_rmse:.4f}")
    
    # 打印一些预测统计信息
    print(f"\n预测统计:")
    print(f"预测值范围: [{submission['tested_positive'].min():.2f}, {submission['tested_positive'].max():.2f}]")
    print(f"预测值均值: {submission['tested_positive'].mean():.2f}")
    print(f"预测值标准差: {submission['tested_positive'].std():.2f}")
    
    print("\n提交文件已保存为 'submission.csv'")

if __name__ == "__main__":
    main()