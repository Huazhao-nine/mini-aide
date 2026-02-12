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

# ==================== 数据加载 ====================
def load_data():
    train_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv'
    test_path = '/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv'
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 确保列顺序一致（除了目标列）
    target_col = 'tested_positive_day3'
    common_cols = [col for col in train_df.columns if col != target_col]
    
    X_train = train_df[common_cols].copy()
    y_train = train_df[target_col].copy()
    X_test = test_df[common_cols].copy()
    
    return X_train, y_train, X_test, test_df['id']

# ==================== 特征工程 ====================
def create_interaction_features(df):
    """创建交互特征"""
    df = df.copy()
    
    # 定义要交互的特征组
    symptom_cols = ['cli', 'ili', 'hh_cmnty_cli', 'nohh_cmnty_cli']
    behavior_cols = ['wearing_mask_7d', 'shop_indoors', 'restaurant_indoors', 'public_transit']
    belief_cols = ['wbelief_masking_effective', 'wbelief_distancing_effective']
    
    # 为每一天创建交互特征
    for day in [1, 2, 3]:
        # 症状 × 行为
        for symptom in symptom_cols:
            for behavior in behavior_cols:
                col1 = f'{symptom}_day{day}'
                col2 = f'w{behavior}_day{day}' if behavior == 'shop_indoors' or behavior == 'restaurant_indoors' else f'{behavior}_day{day}'
                if col1 in df.columns and col2 in df.columns:
                    df[f'inter_{symptom[:3]}_{behavior[:3]}_day{day}'] = df[col1] * df[col2]
        
        # 症状 × 信念
        for symptom in symptom_cols:
            for belief in belief_cols:
                col1 = f'{symptom}_day{day}'
                col2 = f'{belief}_day{day}'
                if col1 in df.columns and col2 in df.columns:
                    df[f'inter_{symptom[:3]}_{belief.split("_")[1][:3]}_day{day}'] = df[col1] * df[col2]
    
    # 创建时间差分特征
    for col in df.columns:
        if 'day' in col and col not in ['id'] + [f'{state}_day{day}' for state in ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 'NM', 'NY', 'NC', 'OH', 'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 'VA', 'WA', 'WV', 'WI'] for day in [1, 2, 3]]:
            base_col = col.replace('_day1', '').replace('_day2', '').replace('_day3', '')
            for d1, d2 in [(1, 2), (2, 3), (1, 3)]:
                col1 = f'{base_col}_day{d1}'
                col2 = f'{base_col}_day{d2}'
                if col1 in df.columns and col2 in df.columns:
                    df[f'diff_{base_col[:10]}_d{d1}d{d2}'] = df[col2] - df[col1]
    
    return df

def engineer_features(X_train, X_test):
    """特征工程主函数"""
    # 保存ID列
    train_id = X_train['id'].copy()
    test_id = X_test['id'].copy()
    
    # 分离州特征和数值特征
    state_cols = [col for col in X_train.columns if col in ['AL', 'AZ', 'CA', 'CO', 'CT', 'FL', 'GA', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MO', 'NJ', 'NM', 'NY', 'NC', 'OH', 'OK', 'OR', 'PA', 'SC', 'TN', 'TX', 'VA', 'WA', 'WV', 'WI']]
    
    # 创建交互特征
    X_train_eng = create_interaction_features(X_train)
    X_test_eng = create_interaction_features(X_test)
    
    # 移除ID列
    X_train_eng = X_train_eng.drop(columns=['id'])
    X_test_eng = X_test_eng.drop(columns=['id'])
    
    return X_train_eng, X_test_eng, train_id, test_id, state_cols

# ==================== 数据预处理 ====================
def preprocess_data(X_train, X_test, y_train, k_features=15):
    """数据标准化和特征选择"""
    # 划分训练集和验证集（最后20%）
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train.iloc[:split_idx]
    X_val_split = X_train.iloc[split_idx:]
    y_train_split = y_train.iloc[:split_idx]
    y_val_split = y_train.iloc[split_idx:]
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val_split)
    X_test_scaled = scaler.transform(X_test)
    
    # 特征选择
    selector = SelectKBest(score_func=f_regression, k=min(k_features, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_split)
    X_val_selected = selector.transform(X_val_scaled)
    X_test_selected = selector.transform(X_test_scaled)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train_selected)
    y_train_tensor = torch.FloatTensor(y_train_split.values).reshape(-1, 1)
    X_val_tensor = torch.FloatTensor(X_val_selected)
    y_val_tensor = torch.FloatTensor(y_val_split.values).reshape(-1, 1)
    X_test_tensor = torch.FloatTensor(X_test_selected)
    
    return (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
            X_test_tensor, selector, scaler)

# ==================== 模型定义 ====================
class COVIDPredictor(nn.Module):
    def __init__(self, input_dim):
        super(COVIDPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
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

# ==================== 训练函数 ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=200, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = []
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # 验证阶段
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # 早停法
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    return model

# ==================== 评估函数 ====================
def evaluate_model(model, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(batch_y.cpu().numpy().flatten())
    
    # 计算RMSE
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 处理负值预测（设置为0）
    predictions = np.maximum(predictions, 0)
    
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    
    score = 1.0 / (1.0 + rmse)
    
    return rmse, score, predictions, actuals

# ==================== 主函数 ====================
def main():
    print("Loading data...")
    X_train, y_train, X_test, test_ids = load_data()
    
    print("Engineering features...")
    X_train_eng, X_test_eng, train_ids, test_ids, state_cols = engineer_features(X_train, X_test)
    
    print("Preprocessing data...")
    (X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, 
     X_test_tensor, selector, scaler) = preprocess_data(X_train_eng, X_test_eng, y_train, k_features=15)
    
    print(f"Selected features: {selector.k_}")
    print(f"Training set shape: {X_train_tensor.shape}")
    print(f"Validation set shape: {X_val_tensor.shape}")
    
    # 创建DataLoader（注意：shuffle=False 对于时间序列数据）
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 初始化模型
    input_dim = X_train_tensor.shape[1]
    model = COVIDPredictor(input_dim)
    
    # 损失函数和优化器
    criterion = nn.L1Loss()  # MAE损失，对异常值更鲁棒
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    print("Training model...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=200, patience=25)
    
    print("Evaluating model...")
    rmse, score, val_preds, val_actuals = evaluate_model(model, val_loader)
    
    # 在测试集上预测
    print("Making predictions on test set...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)
        test_preds = model(X_test_tensor).cpu().numpy().flatten()
    
    # 处理负值预测
    test_preds = np.maximum(test_preds, 0)
    
    # 创建提交文件
    submission = pd.DataFrame({
        'id': test_ids,
        'tested_positive': test_preds
    })
    
    submission_path = 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    # 打印验证分数
    print(f"\nValidation RMSE: {rmse:.6f}")
    print(f"Score= (1.0 / (1.0 + RMSE)) = {score:.6f}")

if __name__ == "__main__":
    main()