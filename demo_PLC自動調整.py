import tkinter as tk
from tkinter import ttk
import os
import pickle
import sys
import numpy as np
import threading
import time
import json
from datetime import datetime
import lightgbm  # 添加這行！



class Config:
    """配置管理類"""

    # 預設系統參數（作為備用）
    DEFAULT_SYSTEM_PARAMS = {
        # 冷卻塔參數
        'wet_bulb_temp': 18.41,  # 濕球溫度 (°C)
        'cooling_water_flow': 200,  # 冷卻水流量 (m³/h)
        'cooling_water_temp': 80,  # 風扇頻率 (%)

        # 冰水機參數
        'chilled_water_temp': 7.0,  # 冰水出水溫度 (°C)
        'cooling_load': 134.67,  # 冷凍噸數 (RT)

        # 泵參數
        'pump_frequency': 85,  # 泵頻率 (%)

        # 溫度限制
        'min_cooling_temp': 26,  # 冷卻水溫度下限 (°C)
        'max_cooling_temp': 33  # 冷卻水溫度上限 (°C)
    }

    # PSO最佳化參數
    PSO_PARAMS = {
        'num_particles': 80,
        'max_iterations': 30,
        'early_stop': 20,
        'min_x11': 3.0,
        'max_x11': 15.0,
        'min_diff': 3.0,
        'max_diff': 5.0
    }

    # 配置檔案讀取間隔（秒）
    CONFIG_READ_INTERVAL = 10

    def __init__(self):
        self.system_params = self.load_system_params()
        self.config_file_path = self._get_config_file_path()

    def _get_config_file_path(self):
        """取得配置檔案路徑"""
        if getattr(sys, 'frozen', False):
            # 打包後的環境 - 配置檔放在 exe 同目錄
            base_path = os.path.dirname(sys.executable)
        else:
            # 開發環境
            base_path = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_path, 'system_config.json')

    def load_system_params(self):
        """從外部JSON檔案載入系統參數"""
        try:
            config_file = self._get_config_file_path()

            # 如果檔案不存在，創建預設檔案
            if not os.path.exists(config_file):
                self.create_default_config_file(config_file)
                print(f"已創建預設配置檔案: {config_file}")

            # 讀取配置檔案
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)

            # 驗證必要參數
            system_params = config_data.get('system_params', {})
            validated_params = self.validate_system_params(system_params)

            return validated_params

        except json.JSONDecodeError as e:
            print(f"配置檔案格式錯誤: {e}")
            print("使用預設參數")
            return self.DEFAULT_SYSTEM_PARAMS.copy()

        except Exception as e:
            print(f"載入配置檔案失敗: {e}")
            print("使用預設參數")
            return self.DEFAULT_SYSTEM_PARAMS.copy()

    def validate_system_params(self, params):
        """驗證並補充缺失的系統參數"""
        validated = self.DEFAULT_SYSTEM_PARAMS.copy()

        for key, default_value in self.DEFAULT_SYSTEM_PARAMS.items():
            if key in params:
                # 驗證數值範圍
                value = params[key]
                if isinstance(value, (int, float)) and value > 0:
                    validated[key] = value
                else:
                    print(f"參數 {key} 值無效，使用預設值: {default_value}")
            else:
                print(f"缺少參數 {key}，使用預設值: {default_value}")

        # 額外驗證溫度邏輯
        if validated['min_cooling_temp'] >= validated['max_cooling_temp']:
            print("溫度範圍設定錯誤，使用預設值")
            validated['min_cooling_temp'] = self.DEFAULT_SYSTEM_PARAMS['min_cooling_temp']
            validated['max_cooling_temp'] = self.DEFAULT_SYSTEM_PARAMS['max_cooling_temp']

        return validated

    def create_default_config_file(self, file_path):
        """創建預設配置檔案"""
        default_config = {
            "system_params": self.DEFAULT_SYSTEM_PARAMS,
            "description": {
                "wet_bulb_temp": "濕球溫度 (°C)",
                "cooling_water_flow": "冷卻水流量 (m³/h)",
                "cooling_water_temp": "冷卻水出水溫度 (°C)",
                "chilled_water_temp": "冰水出水溫度 (°C)",
                "cooling_load": "冷凍噸數 (RT)",
                "pump_frequency": "泵頻率 (hz)",
                "min_cooling_temp": "冷卻水溫度下限 (°C)",
                "max_cooling_temp": "冷卻水溫度上限 (°C)"
            },
            "last_updated": "2025-01-01 00:00:00",
            "version": "1.0"
        }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"創建預設配置檔案失敗: {e}")

    def auto_reload_config(self):
        """自動重新載入配置檔案"""
        try:
            old_params = self.system_params.copy()
            new_params = self.load_system_params()

            # 檢查是否有參數變更
            params_changed = old_params != new_params

            if params_changed:
                self.system_params = new_params
                return True, "配置已更新"
            else:
                return False, "配置無變更"

        except Exception as e:
            return False, f"重載配置失敗: {str(e)}"

    def reload_config(self):
        """手動重新載入配置檔案（保持原有介面）"""
        self.system_params = self.load_system_params()
        return self.system_params

    def save_current_config(self):
        """將當前參數儲存到檔案"""
        try:
            config_file = self.config_file_path

            # 讀取現有檔案以保留其他資訊
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except:
                config_data = {}

            # 更新系統參數
            config_data['system_params'] = self.system_params
            config_data['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')

            # 寫入檔案
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)

            return True

        except Exception as e:
            print(f"儲存配置檔案失敗: {e}")
            return False


class ModelPredictor:
    """模型預測類"""

    def __init__(self):
        self.models = {}
        self.load_models()

    def load_models(self):
        """載入預訓練模型"""
        try:
            # 修改這裡：使用 sys._MEIPASS 而不是 sys.executable
            if getattr(sys, 'frozen', False):
                # 打包後的環境
                base_path = sys._MEIPASS
            else:
                # 開發環境
                base_path = os.path.dirname(os.path.abspath(__file__))

            models_dir = os.path.join(base_path, 'models')

            model_files = {
                'chiller': 'chiller.pkl',
                'tower': 'cooling_tower.pkl',
                'pump': 'pump.pkl'
            }

            for name, filename in model_files.items():
                path = os.path.join(models_dir, filename)
                if os.path.exists(path):
                    try:
                        with open(path, 'rb') as f:
                            self.models[name] = pickle.load(f)
                    except Exception:
                        pass
        except Exception:
            pass

    def predict_chiller(self, cooling_temp, temp_diff, chilled_temp, load):
        """預測冰水機能耗"""
        if 'chiller' in self.models:
            try:
                features = [cooling_temp, temp_diff, chilled_temp, load]
                prediction = self.models['chiller'].predict([features])
                return round(float(prediction[0]), 2)
            except Exception:
                return 0
        return 0

    def predict_tower(self, approach_temp, wet_bulb, flow, fan_freq):
        """預測冷卻塔能耗"""
        if 'tower' in self.models:
            try:
                features = [approach_temp, wet_bulb, flow, fan_freq]
                prediction = self.models['tower'].predict([features])
                return round(float(prediction[0]), 2)
            except Exception:
                return 0
        return 0

    def predict_pump(self, temp_diff, pump_freq, cooling_temp):
        """預測泵能耗"""
        if 'pump' in self.models:
            try:
                features = [temp_diff, pump_freq, cooling_temp]
                prediction = self.models['pump'].predict([features])
                return round(float(prediction[0]), 2)
            except Exception:
                return 0
        return 0

    @property
    def models_loaded(self):
        """檢查模型是否載入完成"""
        required_models = ['chiller', 'tower', 'pump']
        return all(model in self.models for model in required_models)


class PSOOptimizer:
    """粒子群最佳化器"""

    def __init__(self, model_predictor, config):
        self.model_predictor = model_predictor
        self.config = config
        self.pso_params = config.PSO_PARAMS

    def optimize(self):
        """執行PSO最佳化，返回最佳運行參數"""
        if not self.model_predictor.models_loaded:
            return None

        # 使用當前系統參數
        system_params = self.config.system_params

        min_x11 = max(self.pso_params['min_x11'],
                      system_params['min_cooling_temp'] - system_params['wet_bulb_temp'])
        max_x11 = min(self.pso_params['max_x11'],
                      system_params['max_cooling_temp'] - system_params['wet_bulb_temp'])

        # 初始化粒子群
        particles = self._initialize_particles(min_x11, max_x11)
        global_best = {'fitness': float('inf'), 'result': None}

        # 迭代最佳化
        no_improvement = 0
        for _ in range(self.pso_params['max_iterations']):
            improved = False

            for particle in particles:
                # 更新粒子
                self._update_particle(particle, global_best, min_x11, max_x11)

                # 計算適應度
                result = self._calculate_energy(particle['position'][0], particle['position'][1])
                fitness = result['total_energy']

                # 更新最佳解
                if fitness < particle['best_fitness']:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = particle['position'].copy()

                    if fitness < global_best['fitness']:
                        global_best['fitness'] = fitness
                        global_best['result'] = result
                        improved = True

            if improved:
                no_improvement = 0
            else:
                no_improvement += 1

            if no_improvement >= self.pso_params['early_stop']:
                break

        return global_best['result']

    def _initialize_particles(self, min_x11, max_x11):
        """初始化粒子群"""
        particles = []
        for _ in range(self.pso_params['num_particles']):
            position = np.array([
                np.random.uniform(min_x11, max_x11),
                np.random.uniform(self.pso_params['min_diff'], self.pso_params['max_diff'])
            ])
            velocity = np.random.uniform(-0.1, 0.1, 2)

            particles.append({
                'position': position,
                'velocity': velocity,
                'best_position': position.copy(),
                'best_fitness': float('inf')
            })
        return particles

    def _update_particle(self, particle, global_best, min_x11, max_x11):
        """更新粒子速度和位置"""
        if global_best['result'] is not None:
            global_pos = np.array([global_best['result']['x11'], global_best['result']['shared_flow']])

            # 更新速度
            r1, r2 = np.random.rand(2)
            cognitive = 1.5 * r1 * (particle['best_position'] - particle['position'])
            social = 1.5 * r2 * (global_pos - particle['position'])
            particle['velocity'] = 0.7 * particle['velocity'] + cognitive + social

            # 限制速度
            particle['velocity'] = np.clip(particle['velocity'], -0.5, 0.5)

        # 更新位置
        particle['position'] += particle['velocity']
        particle['position'][0] = np.clip(particle['position'][0], min_x11, max_x11)
        particle['position'][1] = np.clip(particle['position'][1],
                                          self.pso_params['min_diff'], self.pso_params['max_diff'])

    def _calculate_energy(self, x11, temp_diff):
        """計算總能耗"""
        system_params = self.config.system_params

        # 計算冷卻水出水溫度
        cooling_temp = np.clip(
            x11 + system_params['wet_bulb_temp'],
            system_params['min_cooling_temp'],
            system_params['max_cooling_temp']
        )

        # 重新計算x11（考慮溫度約束）
        actual_x11 = cooling_temp - system_params['wet_bulb_temp']

        # 預測各設備能耗
        tower_energy = self.model_predictor.predict_tower(
            actual_x11,
            system_params['wet_bulb_temp'],
            system_params['cooling_water_flow'],
            system_params['cooling_water_temp']
        )

        chiller_energy = self.model_predictor.predict_chiller(
            cooling_temp,
            temp_diff,
            system_params['chilled_water_temp'],
            system_params['cooling_load']
        )

        pump_energy = self.model_predictor.predict_pump(
            temp_diff,
            system_params['pump_frequency'],
            cooling_temp
        )

        total_energy = tower_energy + chiller_energy + pump_energy

        return {
            'x11': round(actual_x11, 2),
            'cooling_temp': round(cooling_temp, 2),
            'shared_flow': round(temp_diff, 2),
            'tower_energy': round(tower_energy, 2),
            'chiller_energy': round(chiller_energy, 2),
            'pump_energy': round(pump_energy, 2),
            'total_energy': round(total_energy, 2)
        }


class AutoOptimizerApp:
    """純自動化最佳化應用程式"""

    def __init__(self, root):
        self.root = root
        self.config = Config()
        self.model_predictor = ModelPredictor()
        self.optimizer = PSOOptimizer(self.model_predictor, self.config)

        # 運行狀態
        self.is_running = False
        self.main_thread = None
        self.stop_flag = False

        # 最新結果
        self.latest_result = None

        # 計數器追蹤
        self.cycle_counter = 0

        self.setup_ui()
        self.start_optimization()

    def save_plc_data(self, result):
        """儲存PLC資料"""
        if not result:
            return False

        try:
            data = {
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "unix_timestamp": int(time.time()),
                "control_params": {
                    "approach_temp": result['x11'],
                    "cooling_water_temp": result['cooling_temp'],
                    "cooling_water_diff": result['shared_flow']
                },
                "energy_status": {
                    "total_power": result['total_energy'],
                    "tower_power": result['tower_energy'],
                    "chiller_power": result['chiller_energy'],
                    "pump_power": result['pump_energy'],
                    "plr_ratio": round(result['chiller_energy'] / 2.4, 1)
                },
                "system_params": self.config.system_params,
                "status": "normal"
            }

            # 原子性寫入 - Windows 相容版本
            temp_file = "plc_data_temp.json"
            final_file = "plc_optimization_data.json"

            # 寫入暫存檔案
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Windows 相容的原子性替換
            if os.path.exists(final_file):
                # 如果目標檔案存在，先刪除
                os.remove(final_file)

            # 重新命名暫存檔案
            os.rename(temp_file, final_file)

            return True

        except Exception as e:
            # 清理可能殘留的暫存檔案
            try:
                if os.path.exists("plc_data_temp.json"):
                    os.remove("plc_data_temp.json")
            except:
                pass

            self.log_message(f"儲存PLC資料失敗: {str(e)}")
            return False

    def manual_reload_config(self):
        """手動重新載入配置"""
        try:
            old_params = self.config.system_params.copy()
            new_params = self.config.reload_config()

            if old_params != new_params:
                self.log_message("配置已更新，重新初始化最佳化器")
                self.optimizer = PSOOptimizer(self.model_predictor, self.config)
                self.log_message("配置重載完成")
            else:
                self.log_message("配置無變更")

        except Exception as e:
            self.log_message(f"重載配置失敗: {str(e)}")

    def setup_ui(self):
        """設置簡化UI"""
        self.root.title("冰水系統自動最佳化")
        self.root.geometry("800x550")
        self.root.configure(bg="#f0f0f0")

        # 主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 標題
        title_label = ttk.Label(main_frame, text="AI冰水系統最佳化 - Tkinter介面",
                                font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 20))

        # 狀態顯示
        self.create_status_display(main_frame)

        # 計時器顯示
        self.create_timer_display(main_frame)

        # 結果顯示
        self.create_result_display(main_frame)

        # 控制按鈕
        self.create_control_buttons(main_frame)

        # 日誌顯示
        self.create_log_display(main_frame)

        # 版權資訊
        ttk.Label(self.root, text="© 2025 PLC自動最佳化系統 by terry_中山大學",
                  font=("Arial", 8), background="#f0f0f0").pack(side=tk.BOTTOM, pady=5)

    def create_status_display(self, parent):
        """創建狀態顯示區域"""
        status_frame = ttk.LabelFrame(parent, text="系統狀態", padding="15")
        status_frame.pack(fill=tk.X, pady=(0, 15))

        # 模型狀態
        model_frame = ttk.Frame(status_frame)
        model_frame.pack(fill=tk.X)

        ttk.Label(model_frame, text="模型狀態:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)

        if self.model_predictor.models_loaded:
            self.model_status = ttk.Label(model_frame, text="✓ 已載入", foreground="green",
                                          font=("Arial", 11))
        else:
            self.model_status = ttk.Label(model_frame, text="✗ 未載入", foreground="red",
                                          font=("Arial", 11))
        self.model_status.pack(side=tk.LEFT, padx=(10, 0))

        # 運行狀態
        run_frame = ttk.Frame(status_frame)
        run_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(run_frame, text="運行狀態:", font=("Arial", 11, "bold")).pack(side=tk.LEFT)
        self.run_status = ttk.Label(run_frame, text="準備中...", font=("Arial", 11))
        self.run_status.pack(side=tk.LEFT, padx=(10, 0))

    def create_timer_display(self, parent):
        """創建計時器顯示區域"""
        timer_frame = ttk.LabelFrame(parent, text="執行計時", padding="15")
        timer_frame.pack(fill=tk.X, pady=(0, 15))

        timer_frame.columnconfigure(1, weight=1)

        # 主循環計時器
        ttk.Label(timer_frame, text="下次執行:", font=("Arial", 11, "bold")).grid(
            row=0, column=0, sticky="w", padx=(0, 10))
        self.main_timer_label = ttk.Label(timer_frame, text="0/10秒", font=("Arial", 11))
        self.main_timer_label.grid(row=0, column=1, sticky="w")

        # 執行狀態提示
        ttk.Label(timer_frame, text="執行模式: 讀取特徵 → 最佳化", font=("Arial", 10, "italic")).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(5, 0))

    def create_result_display(self, parent):
        """創建結果顯示區域"""
        result_frame = ttk.LabelFrame(parent, text="最佳化結果", padding="15")
        result_frame.pack(fill=tk.X, pady=(0, 15))

        # 配置網格
        result_frame.columnconfigure(1, weight=1)
        result_frame.columnconfigure(3, weight=1)

        # 結果標籤
        self.result_labels = {}
        results_config = [
            ("冷卻塔趨近溫度:", "x11", "-- °C", 0, 0),
            ("冷卻水出水溫度:", "cooling_temp", "-- °C", 0, 2),
            ("冷卻水溫差:", "shared_flow", "-- °C", 1, 0),
            ("冷卻塔能耗:", "tower_energy", "-- kW", 1, 2),
            ("冰水機能耗:", "chiller_energy", "-- kW", 2, 0),
            ("泵能耗:", "pump_energy", "-- kW", 2, 2),
        ]

        for label_text, key, default_text, row, col in results_config:
            ttk.Label(result_frame, text=label_text, font=("Arial", 10)).grid(
                row=row, column=col, sticky="w", padx=(0, 10), pady=5)
            self.result_labels[key] = ttk.Label(result_frame, text=default_text,
                                                font=("Arial", 10, "bold"))
            self.result_labels[key].grid(row=row, column=col + 1, sticky="w", pady=5)

        # 總能耗（突出顯示）
        ttk.Label(result_frame, text="總能耗:", font=("Arial", 12, "bold")).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(15, 5))
        self.result_labels['total_energy'] = ttk.Label(result_frame, text="-- kW",
                                                       font=("Arial", 14, "bold"), foreground="blue")
        self.result_labels['total_energy'].grid(row=3, column=2, columnspan=2, sticky="w", pady=(15, 5))

    def create_control_buttons(self, parent):
        """創建控制按鈕"""
        button_frame = ttk.Frame(parent)
        button_frame.pack(pady=15)

        self.start_btn = ttk.Button(button_frame, text="停止自動最佳化",
                                    command=self.toggle_optimization, width=20)
        self.start_btn.pack(side=tk.LEFT, padx=10)

        ttk.Button(button_frame, text="手動執行一次",
                   command=self.manual_optimize, width=20).pack(side=tk.LEFT, padx=10)

        ttk.Button(button_frame, text="手動重載配置",
                   command=self.manual_reload_config, width=15).pack(side=tk.LEFT, padx=10)

    def create_log_display(self, parent):
        """創建日誌顯示"""
        log_frame = ttk.LabelFrame(parent, text="運行日誌", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(log_frame, height=8, font=("Arial", 9))
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def log_message(self, message):
        """記錄日誌（執行緒安全）"""

        def _log():
            timestamp = time.strftime('%H:%M:%S')
            formatted_msg = f"[{timestamp}] {message}"
            self.log_text.insert(tk.END, formatted_msg + "\n")
            self.log_text.see(tk.END)

        if threading.current_thread() == threading.main_thread():
            _log()
        else:
            self.root.after(0, _log)

    def update_result_display(self, result):
        """更新結果顯示（執行緒安全）"""

        def _update():
            if result:
                self.result_labels['x11'].config(text=f"{result['x11']:.2f} °C")
                self.result_labels['cooling_temp'].config(text=f"{result['cooling_temp']:.2f} °C")
                self.result_labels['shared_flow'].config(text=f"{result['shared_flow']:.2f} °C")
                self.result_labels['tower_energy'].config(text=f"{result['tower_energy']:.2f} kW")
                self.result_labels['chiller_energy'].config(text=f"{result['chiller_energy']:.2f} kW")
                self.result_labels['pump_energy'].config(text=f"{result['pump_energy']:.2f} kW")
                self.result_labels['total_energy'].config(text=f"{result['total_energy']:.2f} kW")

        if threading.current_thread() == threading.main_thread():
            _update()
        else:
            self.root.after(0, _update)

    def update_timer_display(self, seconds):
        """更新計時器顯示（執行緒安全）"""

        def _update():
            self.main_timer_label.config(text=f"{seconds}/{self.config.CONFIG_READ_INTERVAL}秒")

        if threading.current_thread() == threading.main_thread():
            _update()
        else:
            self.root.after(0, _update)

    def start_optimization(self):
        """啟動自動最佳化"""
        if not self.model_predictor.models_loaded:
            self.log_message("錯誤: 模型未載入完成")
            self.run_status.config(text="✗ 模型載入失敗", foreground="red")
            return

        self.is_running = True
        self.stop_flag = False
        self.run_status.config(text="✓ 自動運行中", foreground="green")
        self.start_btn.config(text="停止自動最佳化")

        # 啟動主執行緒
        self.main_thread = threading.Thread(target=self.main_loop, daemon=True)
        self.main_thread.start()

        self.log_message("自動最佳化已啟動 - 依設定時間自動更新")

    def stop_optimization(self):
        """停止自動最佳化"""
        self.is_running = False
        self.stop_flag = True
        self.run_status.config(text="✗ 已停止", foreground="red")
        self.start_btn.config(text="啟動自動最佳化")
        self.log_message("自動最佳化已停止")

    def toggle_optimization(self):
        """切換最佳化狀態"""
        if self.is_running:
            self.stop_optimization()
        else:
            self.start_optimization()

    def manual_optimize(self):
        """手動執行一次最佳化"""
        if not self.model_predictor.models_loaded:
            self.log_message("錯誤: 模型未載入")
            return

        self.log_message("手動執行最佳化...")
        threading.Thread(target=self.execute_optimization, daemon=True).start()

    def main_loop(self):
        """主循環 - 每10秒執行配置讀取然後最佳化"""
        while not self.stop_flag:
            try:
                # 計時器更新
                for second in range(self.config.CONFIG_READ_INTERVAL):
                    if self.stop_flag:
                        return

                    self.update_timer_display(second + 1)
                    time.sleep(1)

                # 執行配置讀取
                self.execute_config_reload()

                # 配置讀取後立即執行最佳化
                self.execute_optimization()

            except Exception as e:
                self.log_message(f"主循環錯誤: {str(e)}")
                time.sleep(self.config.CONFIG_READ_INTERVAL)

    def execute_config_reload(self):
        """執行配置重載"""
        try:
            changed, message = self.config.auto_reload_config()

            if changed:
                # 配置已變更，需要重新初始化最佳化器
                self.optimizer = PSOOptimizer(self.model_predictor, self.config)
                self.log_message(f"特徵自動更新: {message}")
            else:
                self.log_message("特徵檢查完成 - 無變更")

        except Exception as e:
            self.log_message(f"特徵重載錯誤: {str(e)}")

    def execute_optimization(self):
        """執行一次最佳化"""
        try:
            self.log_message("開始執行最佳化...")
            result = self.optimizer.optimize()

            if result:
                self.latest_result = result
                self.update_result_display(result)
                self.save_plc_data(result)
                self.log_message(f"最佳化完成 - 總能耗: {result['total_energy']:.2f}kW")
            else:
                self.log_message("最佳化失敗: 無法計算結果")

        except Exception as e:
            self.log_message(f"最佳化錯誤: {str(e)}")

    def on_closing(self):
        """關閉程式時的清理"""
        self.stop_flag = True

        # 等待執行緒結束
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=1)

        self.root.destroy()


# 主程式入口點
if __name__ == "__main__":
    root = tk.Tk()
    app = AutoOptimizerApp(root)

    # 設置關閉事件
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    root.mainloop()