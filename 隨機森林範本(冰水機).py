import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import rcParams
import joblib
import time
import datetime
import os
from bayes_opt import BayesianOptimization

# 設定字體
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# 創建模型目錄
model_dir = 'chiller_model_lgbm'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 記錄開始時間
start_time = time.time()

# 加載數據
print("載入數據...")
data = pd.read_csv(r'C:\Users\catyh\Desktop\Data_analyze\冰水機\ALL\train_chiller.csv')

# 選擇特徵和目標變數
X = data[['x1', 'x2', 'x3', 'x4']]
y = data['chiller']

# 處理缺失值
X = X.fillna(X.median())
y = y.fillna(y.median())

# 分割數據為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# 定義MAPE計算函數
def mean_absolute_percentage_error(y_true, y_pred):
    # 避免除以零
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# 定義LGBM評估函數
def lgbm_evaluate(n_estimators, learning_rate, max_depth, num_leaves,
                  subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_samples):
    # 將參數轉換為LGBM可以使用的格式
    params = {
        'n_estimators': int(round(n_estimators)),
        'learning_rate': learning_rate,
        'max_depth': int(round(max_depth)),
        'num_leaves': int(round(num_leaves)),
        'subsample': max(min(subsample, 1), 0),
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'reg_alpha': max(reg_alpha, 0),
        'reg_lambda': max(reg_lambda, 0),
        'min_child_samples': int(round(min_child_samples)),
        'random_state': 42,
        'verbose': -1
    }

    # 創建LGBM模型
    model = LGBMRegressor(**params)

    # 使用交叉驗證評估模型
    cv_results = []

    # 手動實現交叉驗證，以便計算MAPE
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(X_train):
        X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model.fit(X_cv_train, y_cv_train)
        y_cv_pred = model.predict(X_cv_val)

        # 計算MAPE
        mape = mean_absolute_percentage_error(y_cv_val, y_cv_pred)
        cv_results.append(-mape)  # 取負值，因為BayesianOptimization是最大化目標

    return np.mean(cv_results)


# 定義貝葉斯優化的參數範圍
pbounds = {
    'n_estimators': (100, 600),
    'learning_rate': (0.01, 0.2),
    'max_depth': (3, 12),
    'num_leaves': (20, 150),
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'reg_alpha': (0.0, 2.0),
    'reg_lambda': (0.1, 5.0),
    'min_child_samples': (1, 50)
}

# 設置貝葉斯搜索的迭代次數
n_iterations = 20
print(f"將進行 {n_iterations} 次貝葉斯優化...")

# 顯示開始搜索
print("\n開始超參數貝葉斯優化...")
search_start_time = time.time()

# 初始化貝葉斯優化器並執行優化
optimizer = BayesianOptimization(
    f=lgbm_evaluate,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=n_iterations)

# 計算搜索時間
search_time = time.time() - search_start_time
search_time_str = str(datetime.timedelta(seconds=int(search_time)))
print(f"\n超參數搜索完成！耗時: {search_time_str}")

# 獲取最佳超參數
best_params = optimizer.max['params']
# 調整參數格式（確保整數類型）
best_params['n_estimators'] = int(round(best_params['n_estimators']))
best_params['max_depth'] = int(round(best_params['max_depth']))
best_params['num_leaves'] = int(round(best_params['num_leaves']))
best_params['min_child_samples'] = int(round(best_params['min_child_samples']))
best_score = -optimizer.max['target']  # 轉換回MAPE（因為使用的是負MAPE）

print("\n最佳超參數:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"\n最佳MAPE分數: {best_score:.4f}%")

# 使用最佳超參數訓練最終模型
print("\n使用最佳超參數訓練最終模型...")
final_model = LGBMRegressor(**best_params)

# 記錄訓練開始時間
train_start_time = time.time()

# 訓練模型
final_model.fit(X_train, y_train)

# 計算訓練時間
train_time = time.time() - train_start_time

# 記錄預測開始時間
predict_start_time = time.time()

# 訓練集和測試集預測
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

# 計算預測時間
predict_time = time.time() - predict_start_time

# 計算評估指標
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

# 計算總運行時間
total_time = time.time() - start_time
total_time_str = str(datetime.timedelta(seconds=int(total_time)))

# 打印結果
print("\n冰水機模型評估:")
print(f'訓練集 R²: {train_r2:.4f}')
print(f'訓練集 MSE: {train_mse:.4f}')
print(f'訓練集 MAE: {train_mae:.4f}')
print(f'訓練集 MAPE: {train_mape:.4f}%')

print(f'測試集 R²: {test_r2:.4f}')
print(f'測試集 MSE: {test_mse:.4f}')
print(f'測試集 MAE: {test_mae:.4f}')
print(f'測試集 MAPE: {test_mape:.4f}%')

print('\n時間統計：')
print(f'超參數搜索時間: {search_time:.2f} 秒')
print(f'最終模型訓練時間: {train_time:.2f} 秒')
print(f'預測時間: {predict_time:.2f} 秒')
print(f'總運行時間: {total_time_str}')

# 可視化預測結果對比
plt.figure(figsize=(12, 8))
plt.scatter(range(len(y_test)), y_test, label='實際值 (測試集)', alpha=0.6)
plt.scatter(range(len(y_test_pred)), y_test_pred, label='預測值 (測試集)', alpha=0.6)
plt.title(
    f'實際值 vs 預測值 (測試集)\n訓練集 R²: {train_r2:.4f}, 測試集 R²: {test_r2:.4f}, 測試集 MAPE: {test_mape:.4f}%')
plt.xlabel('樣本索引')
plt.ylabel('冰水機度數')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(model_dir, 'chiller_lgbm_prediction.png'), dpi=300, bbox_inches='tight')
plt.show()

# 繪製特徵重要性（以比例方式呈現）
importances = final_model.feature_importances_
# 計算比例
importances_ratio = importances / np.sum(importances) * 100
feature_names = X.columns
# 按重要性排序
indices = np.argsort(importances_ratio)[::-1]  # 反轉排序，最重要的在最上方
sorted_feature_names = [feature_names[i] for i in indices]
sorted_importances = importances_ratio[indices]

plt.figure(figsize=(10, 6))
plt.barh(range(len(sorted_importances)), sorted_importances, tick_label=sorted_feature_names)
plt.xlabel('特徵重要性比例 (%)')
plt.title('特徵重要性比例分析')
# 添加數值標籤
for i, v in enumerate(sorted_importances):
    plt.text(v + 0.5, i, f'{v:.2f}%', va='center')
plt.tight_layout()
plt.grid(True, axis='x')
plt.savefig(os.path.join(model_dir, 'lgbm_feature_importance_ratio.png'), dpi=300, bbox_inches='tight')
plt.show()

# 保存模型
model_path = os.path.join(model_dir, 'chiller.pkl')
joblib.dump(final_model, model_path)
print(f"\n優化後的LGBM模型已保存至: {model_path}")

# 保存貝葉斯優化結果
optimization_results_path = os.path.join(model_dir, 'bayes_optimization_results_lgbm.pkl')
joblib.dump(optimizer, optimization_results_path)
print(f"貝葉斯優化結果已保存至: {optimization_results_path}")

print("\n完成！")