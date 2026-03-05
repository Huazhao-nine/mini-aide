import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
SEED = 42
np.random.seed(SEED)

# 读取数据
train_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/train.csv')
test_df = pd.read_csv('/home/huazhao/DL-HW/ML-2025/mini-aide/input/test.csv')

# 分离特征和目标
TARGET = 'tested_positive_day3'

# 获取特征列：移除id和目标列
feature_cols = [col for col in train_df.columns if col not in ['id', TARGET]]
X = train_df[feature_cols].copy()
y = train_df[TARGET].copy()

# 测试集特征（保持相同列顺序）
X_test = test_df[feature_cols].copy()
test_ids = test_df['id'].copy()

# 简单缺失值处理：用中位数填充
X = X.fillna(X.median())
X_test = X_test.fillna(X_test.median())

# 5折交叉验证设置
n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

# 定义alpha候选值（在0.01到100之间以对数均匀分布）
alpha_values = np.logspace(-2, 2, 20)  # 20个候选值

best_alpha = None
best_score = float('inf')
scores = []

# 对每个alpha进行5折CV评估
for alpha in alpha_values:
    oof_preds = np.zeros(len(X))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        # 划分训练集和验证集
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 创建多项式特征（二阶，包含交互项）
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

        # 只对训练集拟合多项式变换，然后应用到验证集
        X_train_poly = poly.fit_transform(X_train)
        X_val_poly = poly.transform(X_val)

        # 标准化特征
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_poly)
        X_val_scaled = scaler.transform(X_val_poly)

        # 训练模型
        model = Ridge(alpha=alpha, random_state=SEED)
        model.fit(X_train_scaled, y_train)

        # 验证集预测
        val_pred = model.predict(X_val_scaled)
        oof_preds[val_idx] = val_pred

    # 计算总体OOF MSE
    score = mean_squared_error(y, oof_preds)
    scores.append(score)
    
    if score < best_score:
        best_score = score
        best_alpha = alpha
    
    print(f'Alpha={alpha:.4f}: OOF MSE={score:.6f}')

print(f'\nBest alpha: {best_alpha:.4f}')
print(f'Best OOF MSE: {best_score:.6f}')

# 使用最佳alpha重新训练并生成测试集预测
test_preds = np.zeros(len(X_test))

# 存储OOF预测
oof_preds_final = np.zeros(len(X))
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    # 划分训练集和验证集
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 创建多项式特征
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

    # 只对训练集拟合多项式变换
    X_train_poly = poly.fit_transform(X_train)
    X_val_poly = poly.transform(X_val)
    X_test_poly = poly.transform(X_test)

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_val_scaled = scaler.transform(X_val_poly)
    X_test_scaled = scaler.transform(X_test_poly)

    # 使用最佳alpha训练模型
    model = Ridge(alpha=best_alpha, random_state=SEED)
    model.fit(X_train_scaled, y_train)

    # 验证集预测
    val_pred = model.predict(X_val_scaled)
    oof_preds_final[val_idx] = val_pred

    # 计算验证集MSE
    fold_mse = mean_squared_error(y_val, val_pred)
    fold_scores.append(fold_mse)
    print(f'Fold {fold+1}: MSE = {fold_mse:.4f}')

    # 测试集预测（累计）
    test_preds += model.predict(X_test_scaled) / n_folds

# 计算总体OOF MSE
final_mse = mean_squared_error(y, oof_preds_final)
print(f'\nOverall OOF MSE: {final_mse:.6f}')
print(f'FINAL_MSE={final_mse}')

# 生成提交文件
submission = pd.DataFrame({
    'id': test_ids,
    'tested_positive_day3': test_preds
})
submission.to_csv('./working/submission-0.85.csv', index=False)
print('Submission saved to ./working/submission-0.85.csv')