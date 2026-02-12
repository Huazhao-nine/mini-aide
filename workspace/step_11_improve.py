import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 1. 加载数据
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

# 2. 简化但有效的特征工程
def create_smart_features(df, is_train=True):
    """创建关键特征，避免过拟合"""
    df = df.copy()
    
    # 保存id和目标列
    if is_train:
        target_col = 'tested_positive_day3'
        target = df[target_col].copy()
        df = df.drop([target_col], axis=1)
    
    id_col = df['id'] if 'id' in df.columns else None
    if 'id' in df.columns:
        df = df.drop(['id'], axis=1)
    
    # 1. 核心症状特征 - 加权平均值
    for day in [1, 2, 3]:
        if all([f'cli_day{day}' in df.columns, f'ili_day{day}' in df.columns]):
            df[f'symptom_score_day{day}'] = df[f'cli_day{day}'] * 0.7 + df[f'ili_day{day}'] * 0.3
    
    # 2. 行为风险特征
    for day in [1, 2, 3]:
        if all([f'wrestaurant_indoors_day{day}' in df.columns, 
                f'wshop_indoors_day{day}' in df.columns,
                f'wlarge_event_indoors_day{day}' in df.columns]):
            df[f'indoor_risk_day{day}'] = (
                df[f'wrestaurant_indoors_day{day}'] * 0.5 + 
                df[f'wshop_indoors_day{day}'] * 0.3 + 
                df[f'wlarge_event_indoors_day{day}'] * 0.2
            )
    
    # 3. 防护特征
    for day in [1, 2, 3]:
        if f'wearing_mask_7d_day{day}' in df.columns:
            df[f'protection_day{day}'] = df[f'wearing_mask_7d_day{day}']
    
    # 4. 重要时间序列特征 - 只计算关键指标的滞后和变化
    key_features = ['tested_positive', 'cli', 'ili', 'wearing_mask_7d', 'wworried_catch_covid']
    
    for feat in key_features:
        day1_col = f'{feat}_day1'
        day2_col = f'{feat}_day2'
        day3_col = f'{feat}_day3'
        
        if all([col in df.columns for col in [day1_col, day2_col, day3_col]]):
            # 最近变化
            df[f'{feat}_recent_change'] = df[day3_col] - df[day2_col]
            # 两天变化
            df[f'{feat}_two_day_change'] = df[day3_col] - df[day1_col]
            # 平均值
            df[f'{feat}_mean'] = df[[day1_col, day2_col, day3_col]].mean(axis=1)
    
    # 5. 关键交互特征
    if all(['tested_positive_day2' in df.columns, 'symptom_score_day2' in df.columns]):
        df['positivity_symptom_interaction'] = df['tested_positive_day2'] * df['symptom_score_day2'] / 100
    
    if all(['tested_positive_day2' in df.columns, 'protection_day2' in df.columns]):
        df['positivity_protection_interaction'] = df['tested_positive_day2'] * (100 - df['protection_day2']) / 100
    
    # 6. 状态特征
    if all(['tested_positive_day1' in df.columns, 'tested_positive_day2' in df.columns]):
        df['positivity_trend'] = df['tested_positive_day2'] - df['tested_positive_day1']
        
        # 简单的增长因子（避免除以0）
        epsilon = 1e-3
        df['positivity_growth_factor'] = (df['tested_positive_day2'] + epsilon) / (df['tested_positive_day1'] + epsilon)
    
    # 添加id列回数据
    if id_col is not None:
        df['id'] = id_col
    
    # 如果是训练集，添加目标列
    if is_train:
        df['tested_positive_day3'] = target
    
    return df

# 应用特征工程
print("开始智能特征工程...")
train_df_enhanced = create_smart_features(train_df, is_train=True)
test_df_enhanced = create_smart_features(test_df, is_train=False)

print(f"原始特征数: {train_df.shape[1] - 2}")
print(f"增强后特征数: {train_df_enhanced.shape[1] - 2}")

# 3. 分离特征和目标
target_col = 'tested_positive_day3'
X = train_df_enhanced.drop(['id', target_col], axis=1)
y = train_df_enhanced[target_col].values
X_test = test_df_enhanced.drop(['id'], axis=1)

print(f"特征维度: {X.shape}")

# 4. 使用固定验证集（最后20%）
train_size = int(len(X) * 0.8)
X_train, X_val = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

print(f"\n数据划分:")
print(f"  训练集: {len(X_train)} 样本")
print(f"  验证集: {len(X_val)} 样本")
print(f"  测试集: {len(X_test)} 样本")

# 5. 特征选择和标准化
print("\n特征处理...")

# 先标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 使用SelectKBest选择15个最佳特征
k = 15
selector = SelectKBest(score_func=f_regression, k=min(k, X_train_scaled.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

# 获取选择的特征名称
selected_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_indices]

print(f"  原始特征数: {X_train_scaled.shape[1]}")
print(f"  选择后特征数: {X_train_selected.shape[1]}")
print(f"\n选中的top-{k}特征:")
for i, idx in enumerate(selected_indices):
    print(f"  {i+1:2d}. {X.columns[idx]}")

# 6. 定义优化的神经网络模型
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
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
    
    def forward(self, x):
        return self.network(x)

# 7. 训练函数
def train_model(X_train, y_train, X_val, y_val, X_test):
    """训练模型并返回预测结果"""
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # 创建DataLoader
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = COVIDPredictor(input_dim=X_train.shape[1]).to(device)
    
    # 损失函数和优化器 - 使用MAE损失
    criterion = nn.L1Loss()  # MAE损失，对异常值更鲁棒
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=False
    )
    
    # 训练循环
    epochs = 300
    best_val_rmse = float('inf')
    patience = 40
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor.to(device)).cpu().numpy().flatten()
            val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
        
        # 更新学习率
        scheduler.step(val_rmse)
        
        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            best_val_preds = val_preds.copy()
        else:
            patience_counter += 1
        
        if epoch % 50 == 0:
            print(f'  Epoch {epoch}: Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}')
        
        # 早停
        if patience_counter >= patience:
            print(f'  Early stopping at epoch {epoch}')
            break
    
    print(f'  Best Val RMSE: {best_val_rmse:.4f}')
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 在测试集上预测
    model.eval()
    with torch.no_grad():
        test_preds = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    
    return best_val_preds, test_preds, best_val_rmse

# 8. 训练模型
print("\n训练神经网络模型...")
val_preds, test_preds, best_val_rmse = train_model(
    X_train_selected, y_train, 
    X_val_selected, y_val,
    X_test_selected
)

# 9. 计算验证集分数
val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
score = 1.0 / (1.0 + val_rmse)

print(f"\n模型结果:")
print(f"  Val RMSE: {val_rmse:.4f}")
print(f"  Score: {score:.4f}")

# 10. 后处理优化
print("\n后处理优化...")

# 温和的校准：基于验证集分布调整测试集预测
val_mean = np.mean(y_val)
val_std = np.std(y_val)
test_mean = np.mean(test_preds)
test_std = np.std(test_preds)

# 标准化调整：将测试集分布对齐到验证集分布
test_preds_calibrated = (test_preds - test_mean) / test_std * val_std + val_mean

# 确保非负
test_preds_calibrated = np.maximum(test_preds_calibrated, 0)

# 温和的缩尾处理：只处理极端异常值
val_q1 = np.percentile(y_val, 1)
val_q99 = np.percentile(y_val, 99)

# 使用更宽的边界
test_preds_final = np.clip(test_preds_calibrated, val_q1 * 0.8, val_q99 * 1.2)

print(f"  校准前 - 均值: {test_mean:.2f}, 标准差: {test_std:.2f}")
print(f"  校准后 - 均值: {np.mean(test_preds_final):.2f}, 标准差: {np.std(test_preds_final):.2f}")
print(f"  验证集 - 均值: {val_mean:.2f}, 标准差: {val_std:.2f}")

# 11. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_preds_final
})

submission.to_csv('submission.csv', index=False)
print(f"\n提交文件已保存，包含 {len(submission)} 条预测结果")

# 12. 打印最终分数
print(f"\n{'='*60}")
print("最终结果:")
print(f"  Validation RMSE: {val_rmse:.4f}")
print(f"  Score = (1.0 / (1.0 + RMSE)) = {score:.4f}")
print(f"  预测值范围: [{np.min(test_preds_final):.2f}, {np.max(test_preds_final):.2f}]")
print(f"  预测值均值: {np.mean(test_preds_final):.2f}")

print(f"\nScore= (1.0 / (1.0 + RMSE)) = {score:.4f}")