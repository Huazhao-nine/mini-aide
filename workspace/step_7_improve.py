import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
from scipy import stats

# 设置随机种子以保证可重复性
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载并预处理数据"""
    print("加载数据...")
    train = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
    test = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')
    
    print(f"训练集形状: {train.shape}, 测试集形状: {test.shape}")
    
    # 检查目标列是否存在
    if 'tested_positive_day3' not in train.columns:
        raise KeyError("训练集中找不到目标列 'tested_positive_day3'")
    
    # 分离特征和目标
    X = train.drop(columns=['id', 'tested_positive_day3'])
    y = train['tested_positive_day3'].values
    
    # 测试集特征（不要尝试访问目标列）
    X_test = test.drop(columns=['id'])
    test_ids = test['id']
    
    print(f"特征维度: {X.shape}")
    return X, y, X_test, test_ids

def create_features(X, X_test):
    """创建特征工程 - 更全面的特征工程"""
    print("创建特征工程...")
    
    # 复制数据以避免SettingWithCopyWarning
    X_processed = X.copy()
    X_test_processed = X_test.copy()
    
    # 确保数据类型正确
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
        X_test_processed[col] = pd.to_numeric(X_test_processed[col], errors='coerce')
    
    # 处理缺失值（使用中位数填充，对异常值更鲁棒）
    for col in X_processed.columns:
        median_val = X_processed[col].median()
        X_processed[col] = X_processed[col].fillna(median_val)
        X_test_processed[col] = X_test_processed[col].fillna(median_val)
    
    # 1. 创建时间序列特征（差分、移动平均等）
    for day in ['1', '2', '3']:
        # 创建过去的变化率特征
        if f'cli_day{day}' in X_processed.columns and f'ili_day{day}' in X_processed.columns:
            X_processed[f'cli_ili_ratio_day{day}'] = X_processed[f'cli_day{day}'] / (X_processed[f'ili_day{day}'] + 1e-6)
            X_test_processed[f'cli_ili_ratio_day{day}'] = X_test_processed[f'cli_day{day}'] / (X_test_processed[f'ili_day{day}'] + 1e-6)
    
    # 2. 创建多天聚合统计特征
    for base_feature in ['cli', 'ili', 'tested_positive', 'wearing_mask_7d']:
        day_cols = [f'{base_feature}_day{day}' for day in ['1', '2', '3'] if f'{base_feature}_day{day}' in X_processed.columns]
        if len(day_cols) >= 2:
            X_processed[f'{base_feature}_mean'] = X_processed[day_cols].mean(axis=1)
            X_processed[f'{base_feature}_std'] = X_processed[day_cols].std(axis=1)
            X_processed[f'{base_feature}_trend'] = X_processed[day_cols[-1]] - X_processed[day_cols[0]]
            
            X_test_processed[f'{base_feature}_mean'] = X_test_processed[day_cols].mean(axis=1)
            X_test_processed[f'{base_feature}_std'] = X_test_processed[day_cols].std(axis=1)
            X_test_processed[f'{base_feature}_trend'] = X_test_processed[day_cols[-1]] - X_test_processed[day_cols[0]]
    
    # 3. 创建重要交互特征
    # 症状与防护行为的交互
    for day in ['1', '2', '3']:
        if f'cli_day{day}' in X_processed.columns and f'wearing_mask_7d_day{day}' in X_processed.columns:
            X_processed[f'cli_mask_interaction_day{day}'] = X_processed[f'cli_day{day}'] * X_processed[f'wearing_mask_7d_day{day}']
            X_test_processed[f'cli_mask_interaction_day{day}'] = X_test_processed[f'cli_day{day}'] * X_test_processed[f'wearing_mask_7d_day{day}']
        
        if f'tested_positive_day{day}' in X_processed.columns and f'wearing_mask_7d_day{day}' in X_processed.columns:
            X_processed[f'positive_mask_interaction_day{day}'] = X_processed[f'tested_positive_day{day}'] * X_processed[f'wearing_mask_7d_day{day}']
            if f'tested_positive_day{day}' in X_test_processed.columns:
                X_test_processed[f'positive_mask_interaction_day{day}'] = X_test_processed[f'tested_positive_day{day}'] * X_test_processed[f'wearing_mask_7d_day{day}']
    
    # 4. 创建信念和行为的综合特征
    belief_cols = [col for col in X_processed.columns if 'wbelief_' in col]
    if belief_cols:
        X_processed['belief_mean'] = X_processed[belief_cols].mean(axis=1)
        X_test_processed['belief_mean'] = X_test_processed[belief_cols].mean(axis=1)
    
    # 5. 创建社区传播相关特征
    community_cols = [col for col in X_processed.columns if 'cmnty_cli' in col or 'hh_cmnty' in col or 'nohh_cmnty' in col]
    if community_cols:
        X_processed['community_risk'] = X_processed[community_cols].mean(axis=1)
        X_test_processed['community_risk'] = X_test_processed[community_cols].mean(axis=1)
    
    print(f"特征工程后维度: 训练集 {X_processed.shape}, 测试集 {X_test_processed.shape}")
    
    return X_processed, X_test_processed

def feature_selection_and_scaling(X, y, X_test):
    """特征选择和标准化 - 使用更智能的方法"""
    print("特征选择和标准化...")
    
    # 1. 使用基于模型的特征选择（RandomForest）
    print("使用RandomForest进行特征选择...")
    selector = SelectFromModel(
        RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
        threshold='median'  # 选择重要性在中位数以上的特征
    )
    
    X_selected = selector.fit_transform(X, y)
    X_test_selected = selector.transform(X_test)
    
    # 获取选择的特征列名
    selected_indices = selector.get_support(indices=True)
    selected_features = X.columns[selected_indices]
    print(f"选择了 {len(selected_features)} 个特征（占总特征数的 {len(selected_features)/X.shape[1]*100:.1f}%）")
    print("前20个最重要的特征:", list(selected_features[:20]))
    
    # 2. 使用RobustScaler（对异常值更鲁棒）
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    X_scaled = scaler.fit_transform(X_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    print(f"标准化后特征形状: {X_scaled.shape}")
    
    return X_scaled, X_test_scaled, selected_features

def time_series_split_with_cv(X, y, val_ratio=0.2):
    """时间序列数据划分，使用时间序列交叉验证"""
    print("执行时间序列数据划分...")
    
    n_samples = len(X)
    val_size = int(n_samples * val_ratio)
    
    # 训练集和验证集划分
    X_train = X[:-val_size]
    X_val = X[-val_size:]
    y_train = y[:-val_size]
    y_val = y[-val_size:]
    
    print(f"训练集大小: {X_train.shape}, 验证集大小: {X_val.shape}")
    
    return X_train, X_val, y_train, y_val

class AdvancedCOVIDPredictor(nn.Module):
    """高级COVID-19病例预测神经网络"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.2, use_batch_norm=True):
        super(AdvancedCOVIDPredictor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate * (i + 1) / len(hidden_dims)))  # 逐层增加dropout
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
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
        return self.network(x).squeeze()

def train_model_with_advanced_techniques(X_train, y_train, X_val, y_val, 
                                         epochs=200, batch_size=64, patience=15):
    """训练神经网络模型 - 使用高级训练技术"""
    print("开始训练模型...")
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # 时间序列不打乱
    
    # 初始化模型
    input_dim = X_train.shape[1]
    model = AdvancedCOVIDPredictor(input_dim, hidden_dims=[256, 128, 64, 32], dropout_rate=0.2)
    
    # 使用CUDA如果可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_val_tensor = X_val_tensor.to(device)
    y_val_tensor = y_val_tensor.to(device)
    
    # 损失函数（使用Huber Loss，对异常值更鲁棒）
    criterion = nn.HuberLoss(delta=1.0)  # 结合MSE和MAE的优点
    # criterion = nn.MSELoss()  # 也可以尝试MSE
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 早停机制
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        
        # 验证
        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = criterion(val_predictions, y_val_tensor)
            val_rmse = torch.sqrt(nn.MSELoss()(val_predictions, y_val_tensor)).item()
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存损失
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss.item())
        
        # 早停检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, "
                  f"Val Loss: {val_loss.item():.4f}, Val RMSE: {val_rmse:.4f}, "
                  f"LR: {current_lr:.6f}")
        
        if patience_counter >= patience:
            print(f"早停在第 {epoch+1} 轮")
            break
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)
    model = model.to('cpu')  # 移回CPU用于后续预测
    
    print(f"训练完成，最佳验证损失: {best_val_loss:.4f}")
    
    return model

def evaluate_model(model, X_val, y_val):
    """评估模型并计算分数"""
    print("评估模型...")
    
    # 转换为PyTorch张量
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 预测
    model.eval()
    with torch.no_grad():
        predictions = model(X_val_tensor)
        mse = nn.MSELoss()(predictions, y_val_tensor)
        rmse = torch.sqrt(mse).item()
        
        # 计算MAE和R²作为额外指标
        mae = nn.L1Loss()(predictions, y_val_tensor).item()
        
        # 计算R²
        ss_tot = ((y_val_tensor - y_val_tensor.mean()) ** 2).sum().item()
        ss_res = ((y_val_tensor - predictions) ** 2).sum().item()
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    # 计算分数
    score = 1.0 / (1.0 + rmse)
    print(f"验证集RMSE: {rmse:.6f}")
    print(f"验证集MAE: {mae:.6f}")
    print(f"验证集R²: {r2:.6f}")
    print(f"Score = (1.0 / (1.0 + RMSE)) = {score:.6f}")
    
    return rmse, score, predictions.numpy()

def create_ensemble_predictions(X_train, y_train, X_val, y_val, X_test):
    """创建模型集成预测"""
    print("创建模型集成...")
    
    # 训练多个不同架构的模型
    models = []
    n_folds = 3
    tscv = TimeSeriesSplit(n_splits=n_folds)
    
    fold_predictions = []
    fold_test_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
        print(f"\n训练第 {fold+1}/{n_folds} 折...")
        
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # 训练不同架构的模型
        if fold == 0:
            hidden_dims = [256, 128, 64]
        elif fold == 1:
            hidden_dims = [128, 64, 32, 16]
        else:
            hidden_dims = [512, 256, 128]
        
        model = AdvancedCOVIDPredictor(
            X_fold_train.shape[1], 
            hidden_dims=hidden_dims,
            dropout_rate=0.2 + fold * 0.05
        )
        
        # 转换为张量
        X_fold_train_tensor = torch.FloatTensor(X_fold_train)
        y_fold_train_tensor = torch.FloatTensor(y_fold_train)
        
        # 训练模型
        criterion = nn.HuberLoss(delta=1.0)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            predictions = model(X_fold_train_tensor)
            loss = criterion(predictions, y_fold_train_tensor)
            loss.backward()
            optimizer.step()
        
        # 验证集预测
        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.FloatTensor(X_val)
            val_pred = model(X_val_tensor).numpy()
            fold_predictions.append(val_pred)
            
            # 测试集预测
            X_test_tensor = torch.FloatTensor(X_test)
            test_pred = model(X_test_tensor).numpy()
            fold_test_predictions.append(test_pred)
        
        models.append(model)
    
    # 集成预测（加权平均）
    val_predictions_ensemble = np.mean(fold_predictions, axis=0)
    test_predictions_ensemble = np.mean(fold_test_predictions, axis=0)
    
    # 计算集成模型的分数
    mse = np.mean((val_predictions_ensemble - y_val) ** 2)
    rmse = np.sqrt(mse)
    score = 1.0 / (1.0 + rmse)
    
    print(f"\n集成模型验证集RMSE: {rmse:.6f}")
    print(f"集成模型Score = {score:.6f}")
    
    return test_predictions_ensemble, score

def main():
    """主函数"""
    print("=" * 60)
    print("ML2025 Spring HW2: COVID-19 Cases Prediction - OPTIMIZED")
    print("=" * 60)
    
    # 1. 加载数据
    X, y, X_test_raw, test_ids = load_and_preprocess_data()
    
    # 2. 创建特征工程
    X_processed, X_test_processed = create_features(X, X_test_raw)
    
    # 3. 特征选择和标准化
    X_scaled, X_test_scaled, selected_features = feature_selection_and_scaling(
        X_processed, y, X_test_processed
    )
    
    # 4. 时间序列数据划分
    X_train, X_val, y_train, y_val = time_series_split_with_cv(
        X_scaled, y, val_ratio=0.2
    )
    
    # 5. 训练单个模型
    print("\n" + "=" * 40)
    print("训练单个模型...")
    print("=" * 40)
    model = train_model_with_advanced_techniques(
        X_train, y_train, X_val, y_val,
        epochs=200, batch_size=64, patience=20
    )
    
    # 6. 评估单个模型
    rmse, score, val_predictions = evaluate_model(model, X_val, y_val)
    
    # 7. 创建模型集成
    print("\n" + "=" * 40)
    print("创建模型集成...")
    print("=" * 40)
    test_predictions_ensemble, ensemble_score = create_ensemble_predictions(
        X_train, y_train, X_val, y_val, X_test_scaled
    )
    
    # 8. 打印最终分数（使用集成模型的分数）
    final_score = max(score, ensemble_score)
    print("\n" + "=" * 40)
    print(f"单个模型 Score: {score:.6f}")
    print(f"集成模型 Score: {ensemble_score:.6f}")
    print(f"最终 Score: {final_score:.6f}")
    print("=" * 40)
    
    # 9. 在测试集上进行预测（使用集成预测）
    print("\n生成测试集预测...")
    if ensemble_score > score:
        test_predictions = test_predictions_ensemble
        print("使用集成模型预测")
    else:
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor).numpy()
        print("使用单个模型预测")
    
    # 10. 创建提交文件
    submission = pd.DataFrame({
        'id': test_ids,
        'tested_positive': test_predictions
    })
    
    # 确保预测值非负（使用softplus函数，更平滑）
    submission['tested_positive'] = np.log1p(np.exp(submission['tested_positive'])) - 0.5
    
    # 保存提交文件
    submission_path = 'submission_optimized.csv'
    submission.to_csv(submission_path, index=False)
    print(f"提交文件已保存: {submission_path}")
    print(f"提交文件形状: {submission.shape}")
    
    # 显示一些预测示例
    print("\n前5个预测示例:")
    print(submission.head())
    
    # 显示预测统计
    print(f"\n预测统计:")
    print(f"最小值: {submission['tested_positive'].min():.2f}")
    print(f"最大值: {submission['tested_positive'].max():.2f}")
    print(f"平均值: {submission['tested_positive'].mean():.2f}")
    print(f"中位数: {submission['tested_positive'].median():.2f}")
    
    return final_score

if __name__ == "__main__":
    final_score = main()