import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib
import time
import datetime
import json
from bayes_opt import BayesianOptimization

# 輸出 TensorFlow 版本，確認當前使用版本
print(f"TensorFlow 版本: {tf.__version__}")

# 記錄開始時間
start_time = time.time()

# 創建模型目錄
model_dir = 'chiller_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# 載入數據
print("載入數據...")
data = pd.read_csv(r'C:\Users\catyh\Desktop\Data_analyze\冰水機\ALL\train_chiller.csv')
X = data[['x1', 'x2', 'x3', 'x4']]
y = data['chiller']

# 數據預處理
print("數據預處理...")
X = X.fillna(X.median())
y = y.fillna(y.median())
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median())

# 標準化數據
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# 保存StandardScaler對象
joblib.dump(scaler_X, os.path.join(model_dir, 'scaler_X.joblib'))
joblib.dump(scaler_y, os.path.join(model_dir, 'scaler_y.joblib'))

# 分割數據
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)
X_train_opt, X_val, y_train_opt, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 定義簡化後的MLP模型構建和評估函數 - 為貝葉斯優化設計 (改為2層)
def build_and_evaluate_model(neurons_layer1, neurons_layer2,
                             dropout_rate, learning_rate, batch_size):
    # 將參數轉換為適當的類型（貝葉斯優化處理浮點值）
    neurons_layer1 = int(round(neurons_layer1))
    neurons_layer2 = int(round(neurons_layer2))
    batch_size = int(round(batch_size))

    model = Sequential([
        Dense(neurons_layer1, activation='relu', input_dim=X_train_opt.shape[1]),
        Dropout(dropout_rate),
        Dense(neurons_layer2, activation='relu'),
        Dense(1)  # 輸出層
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    # 設置早停
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    # 固定的訓練輪數，配合早停
    epochs = 200
    # 訓練模型
    history = model.fit(
        X_train_opt,
        y_train_opt,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0
    )

    # 在驗證集上評估
    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]

    # 我們希望最大化負MSE（相當於最小化MSE）
    return -val_loss

# 定義貝葉斯優化的參數邊界 - 簡化為2層
pbounds = {
    'neurons_layer1': (32, 256),  # 第一層神經元
    'neurons_layer2': (16, 128),  # 第二層神經元
    'dropout_rate': (0.1, 0.4),   # Dropout率
    'learning_rate': (0.0001, 0.005),  # 學習率
    'batch_size': (16, 64)        # 批次大小
}

# 設置貝葉斯優化
optimizer = BayesianOptimization(
    f=build_and_evaluate_model,
    pbounds=pbounds,
    random_state=42,
    verbose=2
)

# 貝葉斯優化迭代次數
n_iterations = 20
print(f"將執行{n_iterations}次貝葉斯優化迭代...")

# 執行貝葉斯優化
print("\n開始貝葉斯優化進行超參數調整...\n")
search_start_time = time.time()

optimizer.maximize(
    init_points=3,  # 初始隨機探索點數量
    n_iter=n_iterations  # 優化迭代次數
)

search_time = time.time() - search_start_time
search_time_str = search_time  # 記錄為秒數
print(f"\n貝葉斯優化完成！搜索時間: {search_time:.2f} 秒")

# 提取最佳超參數
best_params = optimizer.max['params']
# 將數值參數轉換為整數
best_params['neurons_layer1'] = int(round(best_params['neurons_layer1']))
best_params['neurons_layer2'] = int(round(best_params['neurons_layer2']))
best_params['batch_size'] = int(round(best_params['batch_size']))
best_score = -optimizer.max['target']  # 轉換回MSE（我們使用負MSE來最大化）

print("\n找到的最佳超參數:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"最佳驗證集MSE: {best_score:.6f}")

# 使用最佳超參數在完整訓練集上訓練最終模型
print("\n使用最佳超參數訓練最終模型...")
final_model = Sequential([
    Dense(best_params['neurons_layer1'], activation='relu', input_dim=X_train.shape[1]),
    Dropout(best_params['dropout_rate']),
    Dense(best_params['neurons_layer2'], activation='relu'),
    Dense(1)
])

final_model.compile(
    optimizer=Adam(learning_rate=best_params['learning_rate']),
    loss='mse',
    metrics=['mae']
)

# 設置早停
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# 訓練最終模型
train_start_time = time.time()
final_history = final_model.fit(
    X_train,
    y_train,
    epochs=200,  # 使用固定的訓練輪數配合早停
    batch_size=best_params['batch_size'],
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1  # 顯示進度
)
train_time = time.time() - train_start_time

# 測試集預測時間
predict_start_time = time.time()
y_pred_test_scaled = final_model.predict(X_test, verbose=0)
predict_time = time.time() - predict_start_time

y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled)
y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# 評估最終模型 - 訓練集
y_pred_train_scaled = final_model.predict(X_train, verbose=0)
y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled)
y_train_original = scaler_y.inverse_transform(y_train.reshape(-1, 1))

# 計算評估指標 - 測試集
r2_test = r2_score(y_test_original, y_pred_test)
mse_test = mean_squared_error(y_test_original, y_pred_test)
# 計算MAPE - 測試集（替代原本的RMSE）
mape_test = np.mean(np.abs((y_test_original - y_pred_test) / y_test_original)) * 100
mae_test = mean_absolute_error(y_test_original, y_pred_test)

# 計算評估指標 - 訓練集
r2_train = r2_score(y_train_original, y_pred_train)
mse_train = mean_squared_error(y_train_original, y_pred_train)
# 計算MAPE - 訓練集（替代原本的RMSE）
mape_train = np.mean(np.abs((y_train_original - y_pred_train) / y_train_original)) * 100
mae_train = mean_absolute_error(y_train_original, y_pred_train)

# 顯示精簡的評估結果
print("\n模型評估結果:")
print("\n訓練集 R²: {:.4f}".format(r2_train))
print("訓練集 MSE: {:.4f}".format(mse_train))
print("訓練集 MAPE: {:.4f}%".format(mape_train))
print("訓練集 MAE: {:.4f}".format(mae_train))

print("\n測試集 R²: {:.4f}".format(r2_test))
print("測試集 MSE: {:.4f}".format(mse_test))
print("測試集 MAPE: {:.4f}%".format(mape_test))
print("測試集 MAE: {:.4f}".format(mae_test))

# 時間統計
total_time = time.time() - start_time
print("\n時間統計：")
print("超參數搜索時間: {:.2f} 秒".format(search_time))
print("最終模型訓練時間: {:.2f} 秒".format(train_time))
print("預測時間: {:.2f} 秒".format(predict_time))
print("總運行時間: {}".format(str(datetime.timedelta(seconds=int(total_time)))))

# 保存最佳超參數配置供參考
with open(os.path.join(model_dir, 'best_hyperparameters_2layer.json'), 'w') as f:
    json.dump(best_params, f)

# 保存模型的兩種方法
print("\n保存模型權重...")
final_model.save_weights(os.path.join(model_dir, 'chiller_model_weights.h5'))

# 保存模型為兩種不同格式，提高兼容性
print("保存完整模型 (HDF5格式)...")
final_model.save(os.path.join(model_dir, 'chiller_model.h5'), save_format='h5')

print("保存完整模型 (SavedModel格式)...")
tf.saved_model.save(final_model, os.path.join(model_dir, 'chiller_model_saved'))

# 儲存模型架構為JSON格式，方便後續重建
print("保存模型架構為JSON...")
model_json = final_model.to_json()
with open(os.path.join(model_dir, 'chiller_model_architecture.json'), 'w') as json_file:
    json_file.write(model_json)

print("\n2層貝葉斯優化模型已以多種格式保存到 '{model_dir}' 目錄")
print("\n完成!")