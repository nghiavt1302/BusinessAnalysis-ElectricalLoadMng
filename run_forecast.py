import pandas as pd
import numpy as np
import glob
from tqdm import tqdm
import os
import pmdarima as pm
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Định nghĩa các hằng số
DATA_DIR = r'C:\Users\devtu\Downloads\Data-20260412T143508Z-3-001\Data\output\output'
OUTPUT_DIR = r'C:\Users\devtu\Downloads\drive-download-20260412T134408Z-3-001\Dashboard (tai ca folder nay + run file .pbip trong do)'
COMMON_PERIOD_START = '2012-10-03'

def calc_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero by replacing 0 with a small epsilon
    epsilon = 1e-10
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100

def run_forecast():
    # 1. Đọc và tổng hợp kwh từ 168 file CSV
    print("1. Aggregating kWh data from 168 CSV files...")
    all_files = sorted(glob.glob(os.path.join(DATA_DIR, 'LCL-June2015v2_*.csv')))
    
    aggs = []
    first_file_weather = None
    
    for idx, file in enumerate(tqdm(all_files, desc="Processing CSV files")):
        df = pd.read_csv(file, usecols=['timestamp', 'kwh', 'temp', 'rhum', 'prcp'], parse_dates=['timestamp'])
        
        # 2. Lấy weather data từ file đầu tiên
        if idx == 0:
            first_file_weather = df[['timestamp', 'temp', 'rhum', 'prcp']].drop_duplicates(subset=['timestamp'])
            
        # Group by timestamp -> sum kwh
        agg = df.groupby('timestamp')['kwh'].sum().reset_index()
        aggs.append(agg)
        
        # Giải phóng RAM
        del df
        
    print("Concatenating aggregated data...")
    total_agg = pd.concat(aggs, ignore_index=True)
    # Sum up all files together
    print("Final groupby...")
    final_agg = total_agg.groupby('timestamp')['kwh'].sum().reset_index()
    del total_agg, aggs
    
    print("Processing weather data...")
    # Merge with weather from first file
    final_agg = pd.merge(final_agg, first_file_weather, on='timestamp', how='left')
    
    # 3. Cắt common period
    print(f"3. Filtering common period (>= {COMMON_PERIOD_START})...")
    final_agg = final_agg[final_agg['timestamp'] >= COMMON_PERIOD_START].copy()
    
    # 4. Resample 30min -> hourly
    print("4. Resampling to hourly...")
    final_agg.set_index('timestamp', inplace=True)
    hourly = final_agg.resample('h').agg({
        'kwh': 'sum',
        'temp': 'mean',
        'rhum': 'mean',
        'prcp': 'mean'
    })
    
    # Forward-fill gaps
    hourly.ffill(inplace=True)
    hourly.reset_index(inplace=True)
    
    # 5. Chia Train/Test (7 ngày cuối = 168 giờ)
    print("5. Splitting train/test...")
    TEST_SIZE = 168
    train = hourly.iloc[:-TEST_SIZE].copy()
    test = hourly.iloc[-TEST_SIZE:].copy()
    
    # To avoid MemoryError in SARIMA with very long series (12000+ points),
    # limit training data to the last 1000 hours (approx 40 days) which is sufficient
    MAX_TRAIN_SIZE = 1000
    if len(train) > MAX_TRAIN_SIZE:
        train = train.iloc[-MAX_TRAIN_SIZE:]
        
    print(f"Train size used: {len(train)}, Test size: {len(test)}")
    
    # 6. Huấn luyện ARIMA
    print("6. Training ARIMA (univariate, non-seasonal)...")
    arima_model = pm.auto_arima(train['kwh'], seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, n_jobs=-1)
    
    # 7. Huấn luyện SARIMA
    print("7. Training SARIMA (seasonal=True, m=24)...")
    # Using strict limits to prevent long runtime
    sarima_model = pm.auto_arima(train['kwh'], seasonal=True, m=24, 
                                 max_p=2, max_q=2, max_P=1, max_Q=1, 
                                 trace=True, error_action='ignore', suppress_warnings=True, n_jobs=-1)
                                 
    # 8. Huấn luyện Prophet
    print("8. Training Prophet...")
    prophet_df = train[['timestamp', 'kwh', 'temp']].rename(columns={'timestamp': 'ds', 'kwh': 'y'})
    prophet_model = Prophet()
    prophet_model.add_regressor('temp')
    prophet_model.fit(prophet_df)
    
    # 9. Dự báo 168 giờ test + 24 giờ tương lai
    print("9. Forecasting test and future...")
    FUTURE_SIZE = 24
    
    # ARIMA predictions
    arima_pred, _ = arima_model.predict(n_periods=TEST_SIZE, return_conf_int=True)
    arima_future, _ = arima_model.predict(n_periods=TEST_SIZE + FUTURE_SIZE, return_conf_int=True)
    arima_future = arima_future[-FUTURE_SIZE:]
    
    # SARIMA predictions
    sarima_pred, _ = sarima_model.predict(n_periods=TEST_SIZE, return_conf_int=True)
    sarima_future, _ = sarima_model.predict(n_periods=TEST_SIZE + FUTURE_SIZE, return_conf_int=True)
    sarima_future = sarima_future[-FUTURE_SIZE:]
    
    # Prophet predictions (Test)
    prophet_test_df = test[['timestamp', 'temp']].rename(columns={'timestamp': 'ds'})
    prophet_test_pred = prophet_model.predict(prophet_test_df)
    
    # Future timestamps
    last_timestamp = hourly['timestamp'].max()
    future_timestamps = [last_timestamp + pd.Timedelta(hours=i) for i in range(1, FUTURE_SIZE + 1)]
    
    # Average temp per hour for future forecast
    avg_temp_per_hour = hourly.groupby(hourly['timestamp'].dt.hour)['temp'].mean()
    future_temp = [avg_temp_per_hour[ts.hour] for ts in future_timestamps]
    
    prophet_future_df = pd.DataFrame({'ds': future_timestamps, 'temp': future_temp})
    prophet_future_pred = prophet_model.predict(prophet_future_df)
    
    # 10. Tính Metrics
    print("10. Calculating metrics...")
    actual = test['kwh'].values
    
    metrics = []
    for model_name, pred in [("ARIMA", arima_pred.values), ("SARIMA", sarima_pred.values), ("Prophet", prophet_test_pred['yhat'].values)]:
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mape = calc_mape(actual, pred)
        metrics.append({'model': model_name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape})
        
    metrics_df = pd.DataFrame(metrics)
    print(metrics_df)
    
    # 11, 12, 13. Xuất CSV
    print("11-13. Exporting CSV files...")
    
    results_df = pd.DataFrame({
        'DateHour': test['timestamp'],
        'Actual_kwh': actual,
        'Pred_Arima': arima_pred.values,
        'Pred_Sarima': sarima_pred.values,
        'Pred_Prophet': prophet_test_pred['yhat'].values
    })
    
    future_df = pd.DataFrame({
        'DateHour': future_timestamps,
        'Pred_Arima': arima_future.values,
        'Pred_Sarima': sarima_future.values,
        'Pred_Prophet': prophet_future_pred['yhat'].values,
        'Prophet_Lower_CI': prophet_future_pred['yhat_lower'].values,
        'Prophet_Upper_CI': prophet_future_pred['yhat_upper'].values
    })
    
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'forecast_hourly_results.csv'), index=False)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'forecast_hourly_metrics.csv'), index=False)
    future_df.to_csv(os.path.join(OUTPUT_DIR, 'forecast_hourly_future.csv'), index=False)
    
    print("Done! Files saved successfully.")

if __name__ == "__main__":
    run_forecast()
