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

# 1. 数据加载
def load_data():
    """加载训练集和测试集"""
    train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
    test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')
    
    print(f"训练集形状: {train_df.shape}")
    print(f"测试集形状: {test_df.shape}")
    
    return train_df, test_df

# 2. 数据预处理
def preprocess_data(train_df, test_df):
    """预处理数据，分离特征和目标"""
    # 训练集：分离特征和目标
    target_col = 'tested_positive_day3'
    
    # 检查目标列是否存在
    if target_col not in train_df.columns:
        # 尝试查找正确的目标列名
        possible_targets = [col for col in train_df.columns if 'tested_positive' in col]
        if possible_targets:
            target_col = possible_targets[0]
            print(f"警告: 使用 '{target_col}' 作为目标列")
    
    y_train = train_df[target_col].values
    
    # 删除训练集中的目标列和ID列
    X_train = train_df.drop(columns=['id', target_col])
    
    # 测试集：删除ID列（注意：测试集没有目标列）
    X_test = test_df.drop(columns=['id'])
    
    # 确保训练集和测试集的特征顺序一致
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    print(f"训练特征形状: {X_train.shape}, 目标形状: {y_train.shape}")
    print(f"测试特征形状: {X_test.shape}")
    
    return X_train, y_train, X_test, test_df['id'].values

# 3. 特征工程
def feature_engineering(X_train, X_test, y_train, k_features=15):
    """特征标准化和选择"""
    # 保存特征名用于调试
    feature_names = X_train.columns.tolist()
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 特征选择（仅在训练集上拟合）
    if X_train_scaled.shape[1] > k_features:
        selector = SelectKBest(score_func=f_regression, k=k_features)
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # 获取选择的特征名
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
        print(f"选择了 {len(selected_features)} 个特征")
        print(f"选择的前5个特征: {selected_features[:5]}")
    else:
        X_train_selected = X_train_scaled
        X_test_selected = X_test_scaled
        selected_features = feature_names
        print(f"使用所有 {len(selected_features)} 个特征")
    
    return X_train_selected, X_test_selected, selected_features

# 4. PyTorch模型定义
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout_rate=0.2):
        super(COVIDPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x).squeeze()

# 5. 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, patience=10):
    """训练模型"""
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"早停在第 {epoch+1} 轮")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"轮次 {epoch+1}/{epochs}, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}")
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, best_val_loss

# 6. 主函数
def main():
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 1. 加载数据
    train_df, test_df = load_data()
    
    # 2. 预处理
    X_train, y_train, X_test, test_ids = preprocess_data(train_df, test_df)
    
    # 3. 特征工程
    X_train_selected, X_test_selected, selected_features = feature_engineering(
        X_train, X_test, y_train, k_features=15
    )
    
    # 4. 数据划分（时间序列，不shuffle）
    val_ratio = 0.2
    val_size = int(len(X_train_selected) * val_ratio)
    
    # 按时间顺序划分（最后20%作为验证集）
    X_val = X_train_selected[-val_size:]
    y_val = y_train[-val_size:]
    X_train_final = X_train_selected[:-val_size]
    y_train_final = y_train[:-val_size]
    
    print(f"训练集大小: {X_train_final.shape[0]}, 验证集大小: {X_val.shape[0]}")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_final).to(device)
    y_train_tensor = torch.FloatTensor(y_train_final).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    X_test_tensor = torch.FloatTensor(X_test_selected).to(device)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 时间序列不shuffle
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 5. 初始化模型
    input_dim = X_train_final.shape[1]
    model = COVIDPredictor(input_dim=input_dim, hidden_dims=[64, 32], dropout_rate=0.2).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    print(f"模型输入维度: {input_dim}")
    print(f"模型结构: {model}")
    
    # 6. 训练模型
    print("\n开始训练...")
    model, train_losses, val_losses, best_val_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        epochs=200, patience=15
    )
    
    # 7. 评估模型（在验证集上计算RMSE）
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val_tensor)
        val_mse = criterion(val_predictions, y_val_tensor)
        val_rmse = torch.sqrt(val_mse).item()
    
    # 计算Score（根据竞赛要求）
    score = 1.0 / (1.0 + val_rmse)
    print(f"\n验证集RMSE: {val_rmse:.4f}")
    print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.4f}")
    
    # 8. 在测试集上进行预测
    print("\n在测试集上进行预测...")
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_predictions = test_predictions.cpu().numpy()
    
    # 确保预测值为非负数（病例数不能为负）
    test_predictions = np.maximum(test_predictions, 0)
    
    # 9. 生成提交文件
    submission_df = pd.DataFrame({
        'id': test_ids,
        'tested_positive': test_predictions
    })
    
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"\n提交文件已保存到: {submission_path}")
    print(f"提交文件形状: {submission_df.shape}")
    print(f"预测值范围: [{test_predictions.min():.2f}, {test_predictions.max():.2f}]")
    
    return model, val_rmse, score

if __name__ == "__main__":
    model, val_rmse, score = main()