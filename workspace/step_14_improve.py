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
print("加载数据...")
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

# 2. 简洁高效的特征工程
def create_smart_features(df, is_train=True):
    """创建关键特征，避免过拟合"""
    df = df.copy()
    
    # 保存id和目标列
    if is_train:
        target_col = 'tested_positive_day3'
        target = df[target_col].copy()
        df = df.drop([target_col], axis=1)
    
    id_col = df['id'] if 'id' in df.columns else None
    df = df.drop(['id'], axis=1)
    
    # 核心特征：阳性率趋势（最重要的特征）
    if 'tested_positive_day1' in df.columns and 'tested_positive_day2' in df.columns:
        # 最近两天的变化
        df['positivity_change'] = df['tested_positive_day2'] - df['tested_positive_day1']
        # 变化率
        df['positivity_growth'] = df['positivity_change'] / (df['tested_positive_day1'] + 1e-5)
        # 动量
        df['positivity_momentum'] = df['tested_positive_day2'] + df['positivity_change']
    
    # 症状相关特征聚合
    for day in [1, 2, 3]:
        if f'cli_day{day}' in df.columns:
            # 症状严重程度指标
            df[f'symptom_intensity_day{day}'] = df[f'cli_day{day}'] + df[f'ili_day{day}']
    
    # 防护与风险交互特征
    for day in [1, 2, 3]:
        # 风险行为指数
        risk_cols = []
        if f'wrestaurant_indoors_day{day}' in df.columns:
            risk_cols.append(f'wrestaurant_indoors_day{day}')
        if f'wshop_indoors_day{day}' in df.columns:
            risk_cols.append(f'wshop_indoors_day{day}')
        
        if risk_cols:
            df[f'risk_behavior_day{day}'] = df[risk_cols].mean(axis=1)
            
        # 防护有效性（口罩使用 * 口罩信念）
        if f'wearing_mask_7d_day{day}' in df.columns and f'wbelief_masking_effective_day{day}' in df.columns:
            df[f'protection_power_day{day}'] = df[f'wearing_mask_7d_day{day}'] * df[f'wbelief_masking_effective_day{day}'] / 100
    
    # 社区传播指标
    for day in [1, 2, 3]:
        if f'hh_cmnty_cli_day{day}' in df.columns and f'nohh_cmnty_cli_day{day}' in df.columns:
            df[f'community_spread_day{day}'] = df[f'hh_cmnty_cli_day{day}'] * 0.7 + df[f'nohh_cmnty_cli_day{day}'] * 0.3
    
    # 疫苗接种影响
    for day in [1, 2, 3]:
        if f'wcovid_vaccinated_friends_day{day}' in df.columns and f'wworried_catch_covid_day{day}' in df.columns:
            df[f'vax_confidence_day{day}'] = df[f'wcovid_vaccinated_friends_day{day}'] / (df[f'wworried_catch_covid_day{day}'] + 1)
    
    # 添加id列回数据
    if id_col is not None:
        df['id'] = id_col
    
    # 如果是训练集，添加目标列
    if is_train:
        df['tested_positive_day3'] = target
    
    return df

# 应用特征工程
print("特征工程...")
train_df_enhanced = create_smart_features(train_df, is_train=True)
test_df_enhanced = create_smart_features(test_df, is_train=False)

print(f"特征数量: {train_df_enhanced.shape[1] - 2}")

# 3. 分离特征和目标
target_col = 'tested_positive_day3'
X = train_df_enhanced.drop(['id', target_col], axis=1)
y = train_df_enhanced[target_col].values
X_test = test_df_enhanced.drop(['id'], axis=1)

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

# 使用StandardScaler（按照任务要求）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 使用SelectKBest选择15个最重要的特征（按照任务建议）
k = min(15, X_train.shape[1])  # 确保k不超过特征数
selector = SelectKBest(score_func=f_regression, k=k)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

print(f"  原始特征数: {X_train.shape[1]}")
print(f"  选择后特征数: {X_train_selected.shape[1]}")

# 获取选择的特征名称
selected_features = X.columns[selector.get_support()]
print(f"\n最重要的特征:")
for i, feat in enumerate(selected_features[:10]):  # 只显示前10个
    print(f"  {i+1}. {feat}")

# 6. 定义改进的神经网络模型
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# 7. 训练函数
def train_model(X_train, y_train, X_val, y_val, X_test):
    """训练模型并评估"""
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # 创建DataLoader（注意：shuffle=False，按照时间序列要求）
    batch_size = 64
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = COVIDPredictor(input_dim=X_train.shape[1]).to(device)
    
    # 使用L1损失（MAE）进行训练
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=False
    )
    
    # 训练循环
    epochs = 300
    best_val_rmse = float('inf')
    patience = 25
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
            
            train_loss += loss.item() * batch_X.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor.to(device)).cpu().numpy()
            val_rmse = np.sqrt(np.mean((val_preds - y_val) ** 2))
        
        # 更新学习率
        scheduler.step(val_rmse)
        
        # 保存最佳模型
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        # 打印进度
        if epoch % 30 == 0:
            print(f'  Epoch {epoch}: Train MAE: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}')
        
        # 早停
        if patience_counter >= patience:
            print(f'  Early stopping at epoch {epoch}')
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 在验证集上做最终预测
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_tensor.to(device)).cpu().numpy().flatten()
    
    # 在测试集上预测
    with torch.no_grad():
        test_preds = model(X_test_tensor.to(device)).cpu().numpy().flatten()
    
    return val_preds, test_preds, best_val_rmse

# 8. 训练模型
print("\n训练神经网络模型...")
val_preds, test_preds, best_val_rmse = train_model(
    X_train_selected, y_train, 
    X_val_selected, y_val,
    X_test_selected
)

# 9. 后处理优化
print("\n后处理优化...")

# 确保非负（重要！）
test_preds = np.maximum(test_preds, 0)

# 基于验证集分布进行温和调整
val_preds_adj = np.maximum(val_preds, 0)
val_rmse = np.sqrt(np.mean((val_preds_adj - y_val) ** 2))
score = 1.0 / (1.0 + val_rmse)

print(f"\n模型性能:")
print(f"  验证集RMSE: {val_rmse:.4f}")
print(f"  Score: {score:.4f}")

# 10. 生成提交文件
submission = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive': test_preds
})

submission.to_csv('submission.csv', index=False)
print(f"\n提交文件已保存，包含 {len(submission)} 条预测结果")

# 11. 打印最终分数
print(f"\n{'='*60}")
print(f"最终结果:")
print(f"  验证集预测均值: {val_preds_adj.mean():.2f}, 真实均值: {y_val.mean():.2f}")
print(f"  测试集预测均值: {test_preds.mean():.2f}")
print(f"  验证集预测标准差: {val_preds_adj.std():.2f}, 真实标准差: {y_val.std():.2f}")
print(f"  测试集预测标准差: {test_preds.std():.2f}")

print(f"\nScore= (1.0 / (1.0 + RMSE)) = {score:.4f}")