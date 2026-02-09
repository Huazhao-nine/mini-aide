import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可重复性
np.random.seed(42)
torch.manual_seed(42)

# ========== 1. 数据加载与预处理 ==========
def load_and_preprocess_data():
    # 加载数据
    train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
    test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    # 分离特征和目标
    TARGET = 'tested_positive_day3'
    
    # 保存测试集ID用于提交
    test_ids = test_df['id'].copy()
    
    # 移除id列
    X_train = train_df.drop(['id', TARGET], axis=1)
    y_train = train_df[TARGET].copy()
    
    X_test = test_df.drop(['id'], axis=1)
    
    return X_train, y_train, X_test, test_ids

# ========== 2. 特征工程 ==========
def feature_engineering(X_train, y_train, X_test, k_features=20):
    # 创建交互特征（可选）
    interaction_features = []
    
    # 示例：创建一些可能有意义的交互特征
    # 这里只是示例，可以根据实际特征含义调整
    if 'cli_day1' in X_train.columns and 'wearing_mask_7d_day1' in X_train.columns:
        X_train['cli_mask_interaction_day1'] = X_train['cli_day1'] * X_train['wearing_mask_7d_day1']
        X_test['cli_mask_interaction_day1'] = X_test['cli_day1'] * X_test['wearing_mask_7d_day1']
        interaction_features.append('cli_mask_interaction_day1')
    
    if 'cli_day2' in X_train.columns and 'wearing_mask_7d_day2' in X_train.columns:
        X_train['cli_mask_interaction_day2'] = X_train['cli_day2'] * X_train['wearing_mask_7d_day2']
        X_test['cli_mask_interaction_day2'] = X_test['cli_day2'] * X_test['wearing_mask_7d_day2']
        interaction_features.append('cli_mask_interaction_day2')
    
    print(f"创建了 {len(interaction_features)} 个交互特征")
    
    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 特征选择
    selector = SelectKBest(score_func=f_regression, k=min(k_features, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # 获取选择的特征名
    selected_features = X_train.columns[selector.get_support()]
    print(f"选择了 {len(selected_features)} 个最佳特征")
    print(f"选择的特征: {list(selected_features)}")
    
    return X_train_selected, X_test_selected, y_train.values

# ========== 3. 按时间顺序划分验证集 ==========
def time_series_split(X, y, val_ratio=0.2):
    """
    按时间顺序划分验证集（最后20%作为验证集）
    """
    n_samples = len(X)
    val_size = int(n_samples * val_ratio)
    
    # 按顺序划分：训练集 = 前80%，验证集 = 后20%
    X_train = X[:-val_size]
    y_train = y[:-val_size]
    X_val = X[-val_size:]
    y_val = y[-val_size:]
    
    print(f"训练集大小: {len(X_train)}, 验证集大小: {len(X_val)}")
    return X_train, X_val, y_train, y_val

# ========== 4. 神经网络模型 ==========
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

# ========== 5. 训练函数 ==========
def train_model(model, X_train, y_train, X_val, y_val, epochs=100, lr=0.001):
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)
    
    # 数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 训练循环
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)
        val_losses.append(val_loss.item())
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss.item():.4f}')
    
    return train_losses, val_losses

# ========== 6. 评估函数 ==========
def evaluate_model(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.FloatTensor(X_val)
        predictions = model(X_val_tensor).numpy().flatten()
    
    # 计算RMSE
    mse = np.mean((predictions - y_val) ** 2)
    rmse = np.sqrt(mse)
    
    # 计算Score
    score = 1.0 / (1.0 + rmse)
    
    print(f"验证集RMSE: {rmse:.4f}")
    print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.4f}")
    
    return rmse, score

# ========== 7. 主程序 ==========
def main():
    # 1. 加载数据
    print("步骤1: 加载数据...")
    X_train_raw, y_train, X_test_raw, test_ids = load_and_preprocess_data()
    
    # 2. 特征工程
    print("\n步骤2: 特征工程...")
    X_train_processed, X_test_processed, y_train_processed = feature_engineering(
        X_train_raw, y_train, X_test_raw, k_features=20
    )
    
    # 3. 按时间顺序划分验证集
    print("\n步骤3: 划分训练集和验证集...")
    X_train, X_val, y_train, y_val = time_series_split(
        X_train_processed, y_train_processed, val_ratio=0.2
    )
    
    # 4. 创建和训练模型
    print("\n步骤4: 创建和训练模型...")
    input_dim = X_train.shape[1]
    model = COVIDPredictor(input_dim)
    
    print(f"模型输入维度: {input_dim}")
    print(f"模型结构:\n{model}")
    
    # 训练模型
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_val, y_val, 
        epochs=80, lr=0.001
    )
    
    # 5. 评估模型
    print("\n步骤5: 评估模型...")
    rmse, score = evaluate_model(model, X_val, y_val)
    
    # 6. 在测试集上进行预测
    print("\n步骤6: 生成提交文件...")
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test_processed)
        test_predictions = model(X_test_tensor).numpy().flatten()
    
    # 确保预测值为非负数（病例数不能为负）
    test_predictions = np.maximum(test_predictions, 0)
    
    # 7. 创建提交文件
    submission_df = pd.DataFrame({
        'id': test_ids,
        'tested_positive': test_predictions
    })
    
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"提交文件已保存到: {submission_path}")
    print(f"提交文件形状: {submission_df.shape}")
    print(f"预测范围: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")

if __name__ == "__main__":
    main()