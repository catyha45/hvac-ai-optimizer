# HVAC AI Optimizer - 冰水系統智慧最佳化

使用粒子群最佳化（PSO）演算法和機器學習模型，自動最佳化冰水系統的能源效率。

## 系統需求
- Python 3.8+
- tkinter（通常內建）
- numpy >= 1.21.0
- lightgbm >= 3.3.0

## 安裝說明
1. 安裝套件：
   ```bash
   pip install -r requirements.txt
   ```

2. **重要**：需要自行訓練模型
   - 本專案不包含預訓練模型（基於資料保護考量）
   - 請使用 `隨機森林範本(冰水機).py` 訓練模型
   - 將訓練好的 .pkl 檔案放入 `models/` 資料夾

## 使用方法
```bash
python demo_PLC自動調整.py
```

## 專案結構
```
hvac-ai-optimizer/
├── demo_PLC自動調整.py         # 主程式
├── 隨機森林範本(冰水機).py      # 隨機森林模型訓練腳本
├── 神經網路範本(冰水機).py      # 神經網路模型訓練腳本
├── models/                     # AI模型檔案資料夾
│   └── README.txt             # 模型說明檔案
├── system_config.json          # 系統參數配置
├── requirements.txt            # 套件相依
└── README.md                  # 專案說明
```

## 主要功能
- 自動讀取系統參數配置
- PSO演算法進行最佳化計算
- 即時能耗預測與監控
- Tkinter GUI 操作介面
- 自動儲存最佳化結果為 JSON 格式

## 技術特色
- 整合冰水機、冷卻塔、泵的能耗預測模型
- 支援參數動態調整與配置熱重載
- 多執行緒架構確保 UI 響應性
- 完整的錯誤處理與日誌記錄

## 模型重新訓練
如需重新訓練冰水機模型：
```bash
python 隨機森林範本(冰水機).py
```

產生的新模型會儲存為 `models/chiller.pkl`

## 開發者
terry_中山大學

## 版本
v1.0 - 初版功能完成
