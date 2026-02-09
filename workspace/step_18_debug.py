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

# 设置随机种子以确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 数据加载
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")

# 2. 分离特征和目标
# 确认目标列存在
target_col = 'tested_positive_day3'
if target_col not in train_df.columns:
    # 如果列名不匹配，尝试找到正确的目标列
    possible_targets = [col for col in train_df.columns if 'tested_positive' in col]
    if possible_targets:
        target_col = possible_targets[-1]  # 取最后一个包含'tested_positive'的列
        print(f"目标列调整为: {target_col}")
    else:
        raise ValueError("未找到目标列'tested_positive_day3'或类似列")

# 分离训练集的特征和目标
y_train = train_df[target_col].values
X_train = train_df.drop(columns=['id', target_col])

# 保存测试集ID用于提交
test_ids = test_df['id'].values
X_test = test_df.drop(columns=['id'])

print(f"训练特征形状: {X_train.shape}, 训练目标形状: {y_train.shape}")
print(f"测试特征形状: {X_test.shape}")

# 3. 时间序列划分训练集和验证集（禁止打乱）
val_size = int(len(X_train) * 0.2)  # 使用最后20%作为验证集

# 按时间顺序划分
X_val = X_train.iloc[-val_size:].copy()
y_val = y_train[-val_size:].copy()

X_train = X_train.iloc[:-val_size].copy()
y_train = y_train[:-val_size].copy()

print(f"划分后 - 训练集: {X_train.shape}, 验证集: {X_val.shape}")

# 4. 特征工程
# 数据标准化（使用训练集拟合）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 特征选择（使用训练集选择特征）
k = 15  # 选择15个最佳特征
selector = SelectKBest(score_func=f_regression, k=k)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_val_selected = selector.transform(X_val_scaled)
X_test_selected = selector.transform(X_test_scaled)

print(f"特征选择后 - 训练集: {X_train_selected.shape}, 验证集: {X_val_selected.shape}")

# 5. 转换为PyTorch张量
X_train_tensor = torch.FloatTensor(X_train_selected)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_val_tensor = torch.FloatTensor(X_val_selected)
y_val_tensor = torch.FloatTensor(y_val).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_selected)

# 创建数据加载器
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 时间序列，不洗牌

# 6. 定义DNN模型
class COVIDPredictor(nn.Module):
    def __init__(self, input_size):
        super(COVIDPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# 初始化模型
input_size = X_train_selected.shape[1]
model = COVIDPredictor(input_size)
print(f"模型输入维度: {input_size}")

# 7. 训练配置
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# 8. 训练模型
num_epochs = 100
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_losses = []
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_loss = criterion(val_predictions, y_val_tensor)
    
    scheduler.step(val_loss)
    
    # 早停检查
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        best_model_state = model.state_dict().copy()
        patience_counter = 0
    else:
        patience_counter += 1
    
    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss.item():.4f}')
    
    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch+1}')
        model.load_state_dict(best_model_state)
        break

# 9. 在验证集上评估
model.eval()
with torch.no_grad():
    val_predictions = model(X_val_tensor)
    val_mse = criterion(val_predictions, y_val_tensor).item()
    val_rmse = np.sqrt(val_mse)

# 计算评估分数
score = 1.0 / (1.0 + val_rmse)
print(f"\n验证集评估:")
print(f"MSE: {val_mse:.4f}")
print(f"RMSE: {val_rmse:.4f}")
print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.4f}")

# 10. 对测试集进行预测
model.eval()
with torch.no_grad():
    test_predictions = model(X_test_tensor)

# 确保预测值非负（病例数不能为负）
test_predictions_np = test_predictions.numpy().flatten()
test_predictions_np = np.maximum(test_predictions_np, 0)

# 11. 生成提交文件
submission_df = pd.DataFrame({
    'id': test_ids,
    'tested_positive': test_predictions_np
})

# 保存提交文件
submission_df.to_csv('submission.csv', index=False)
print(f"\n提交文件已保存: submission.csv")
print(f"提交文件形状: {submission_df.shape}")
print(f"预测值范围: [{test_predictions_np.min():.2f}, {test_predictions_np.max():.2f}]")

# 12. 额外：特征重要性分析（可选）
feature_scores = selector.scores_
selected_features_indices = selector.get_support(indices=True)
selected_features = X_train.columns[selected_features_indices]

print(f"\nTop {k}个重要特征:")
for i, (idx, score) in enumerate(zip(selected_features_indices, feature_scores[selected_features_indices])):
    feature_name = X_train.columns[idx] if idx < len(X_train.columns) else f"Feature_{idx}"
    print(f"{i+1}. {feature_name}: {score:.2f}")