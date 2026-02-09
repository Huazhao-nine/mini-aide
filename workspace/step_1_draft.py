import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以保证可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==================== 数据加载 ====================
print("正在加载数据...")
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

# 保存测试集ID用于最终提交
test_ids = test_df['id'].copy()

# 分离特征和目标
target_col = 'tested_positive_day3'
y_train = train_df[target_col].values

# 删除ID列和目标列
X_train = train_df.drop(columns=['id', target_col])
X_test = test_df.drop(columns=['id'])

# 确保训练集和测试集特征顺序一致
assert list(X_train.columns) == list(X_test.columns), "特征列不匹配"

# ==================== 特征工程 ====================
print("正在进行特征工程...")

# 1. 创建交互特征（示例：与戴口罩相关的交互）
def create_interaction_features(df):
    """创建交互特征"""
    df = df.copy()
    
    # 示例交互特征：症状指标 * 防护行为
    for day in ['1', '2', '3']:
        if f'cli_day{day}' in df.columns and f'wearing_mask_7d_day{day}' in df.columns:
            df[f'cli_mask_interaction_day{day}'] = df[f'cli_day{day}'] * df[f'wearing_mask_7d_day{day}']
        
        if f'ili_day{day}' in df.columns and f'wearing_mask_7d_day{day}' in df.columns:
            df[f'ili_mask_interaction_day{day}'] = df[f'ili_day{day}'] * df[f'wearing_mask_7d_day{day}']
        
        if f'cli_day{day}' in df.columns and f'wbelief_masking_effective_day{day}' in df.columns:
            df[f'cli_mask_belief_interaction_day{day}'] = df[f'cli_day{day}'] * df[f'wbelief_masking_effective_day{day}']
    
    # 跨天的特征聚合（简单求和）
    for prefix in ['cli_day', 'ili_day', 'tested_positive_day']:
        day_cols = [col for col in df.columns if col.startswith(prefix)]
        if len(day_cols) >= 2:
            df[f'{prefix.rstrip("day_")}total'] = df[day_cols].sum(axis=1)
    
    return df

# 应用特征工程
X_train_fe = create_interaction_features(X_train)
X_test_fe = create_interaction_features(X_test)

# ==================== 数据预处理 ====================
# 使用训练集拟合scaler，然后转换所有数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_fe)
X_test_scaled = scaler.transform(X_test_fe)

# 特征选择
print("正在进行特征选择...")
k_best = 15  # 选择15个最佳特征
selector = SelectKBest(score_func=f_regression, k=min(k_best, X_train_scaled.shape[1]))
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

print(f"原始特征数: {X_train_scaled.shape[1]}")
print(f"选择后特征数: {X_train_selected.shape[1]}")

# ==================== 时间序列数据划分 ====================
print("正在进行时间序列数据划分...")
val_size = int(0.2 * len(X_train_selected))  # 最后20%作为验证集

# 按时间顺序划分（禁止打乱！）
X_val = X_train_selected[-val_size:]
y_val = y_train[-val_size:]

X_tr = X_train_selected[:-val_size]
y_tr = y_train[:-val_size]

print(f"训练集大小: {X_tr.shape[0]}")
print(f"验证集大小: {X_val.shape[0]}")

# ==================== PyTorch数据准备 ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 转换为PyTorch张量
X_tr_tensor = torch.FloatTensor(X_tr).to(device)
y_tr_tensor = torch.FloatTensor(y_tr).reshape(-1, 1).to(device)

X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1).to(device)

X_test_tensor = torch.FloatTensor(X_test_selected).to(device)

# 创建DataLoader
batch_size = 64
train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # 训练时可以打乱

# ==================== 定义神经网络模型 ====================
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.model(x)

# 初始化模型
model = COVIDPredictor(input_dim=X_tr.shape[1]).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# ==================== 训练模型 ====================
print("开始训练模型...")
epochs = 150
best_val_loss = float('inf')
patience_counter = 0
patience_limit = 20

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_loss = criterion(val_predictions, y_val_tensor).item()
    
    # 学习率调度
    scheduler.step(val_loss)
    
    # 早停检查
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
    
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if patience_counter >= patience_limit:
        print(f"早停触发于第 {epoch} 轮")
        break

# 加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))

# ==================== 在验证集上评估 ====================
model.eval()
with torch.no_grad():
    val_predictions = model(X_val_tensor)
    val_mse = criterion(val_predictions, y_val_tensor).item()
    val_rmse = np.sqrt(val_mse)

# 计算分数
score = 1.0 / (1.0 + val_rmse)
print(f"\n验证集评估结果:")
print(f"MSE: {val_mse:.6f}")
print(f"RMSE: {val_rmse:.6f}")
print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.6f}")

# 打印要求的格式
print(f"\nScore: {score}")

# ==================== 生成测试集预测 ====================
print("生成测试集预测...")
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)
    test_predictions_np = test_predictions.cpu().numpy().flatten()

# 确保没有负值预测（病例数不能为负）
test_predictions_np = np.maximum(test_predictions_np, 0)

# ==================== 生成提交文件 ====================
submission = pd.DataFrame({
    'id': test_ids,
    'tested_positive': test_predictions_np
})

# 保存提交文件
submission.to_csv('submission.csv', index=False)
print(f"提交文件已保存为 'submission.csv'")
print(f"预测值范围: {test_predictions_np.min():.2f} - {test_predictions_np.max():.2f}")

# 显示前几行预测结果
print("\n提交文件前5行:")
print(submission.head())