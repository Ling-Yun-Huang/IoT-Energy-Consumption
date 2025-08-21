## IoT-Based Energy Consumption Analysis and Forecasting for Building Systems


***

**Project Owner:** Ling-Yun, Huang

**Skills:** 

***


```python
# Data handling & visualization
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Data Split
from sklearn.model_selection import TimeSeriesSplit

# Machine learning models
from sklearn.ensemble import RandomForestRegressor

# Gradient boosting frameworks
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

# LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam

# Evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Time series forecasting
#from statsmodels.tsa.arima.model import ARIMA
#from prophet import Prophet
```

## Data Preprossing


```python
# Import datasets
EC1 = pd.read_csv("KwhConsumptionBlower78_1.csv")
EC2 = pd.read_csv("KwhConsumptionBlower78_2.csv")
EC3 = pd.read_csv("KwhConsumptionBlower78_3.csv")

# Combine Datasets
raw_data = pd.concat([EC1,EC2,EC3], ignore_index=True)
print(raw_data.head())
print(raw_data.info())
df = raw_data.copy()
```

       Unnamed: 0      TxnDate   TxnTime  Consumption
    0       76229  01 Jan 2022  16:55:52        1.010
    1       76258  01 Jan 2022  21:45:29        0.908
    2       76287  01 Jan 2022  12:24:52        0.926
    3       76316  01 Jan 2022  04:07:36        0.928
    4       76345  01 Jan 2022  06:52:25        0.916
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3606 entries, 0 to 3605
    Data columns (total 4 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   Unnamed: 0   3606 non-null   int64  
     1   TxnDate      3606 non-null   object 
     2   TxnTime      3606 non-null   object 
     3   Consumption  3606 non-null   float64
    dtypes: float64(1), int64(1), object(2)
    memory usage: 112.8+ KB
    None



```python
# The statistical summary of Energy Consumption
df['Consumption'].describe()
```




    count    3606.000000
    mean        2.781508
    std         2.961248
    min         0.000000
    25%         0.948000
    50%         1.032000
    75%         4.005500
    max        27.284000
    Name: Consumption, dtype: float64




```python
## Feature Engineering
# Create a datetime column from 'TxnDate' and 'TxnTime'
df['datetime'] = pd.to_datetime(df['TxnDate'] + ' ' + df['TxnTime'], format='%d %b %Y %H:%M:%S')
df = df.sort_values('datetime').reset_index(drop=True) # sort by datetime

# Extract time-based features
df['hour_of_day'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Compute rolling statistics with 1-hour window
df = df.set_index('datetime')
df['rolling_mean_1h'] = df['Consumption'].rolling('1h', closed='left').mean()
df['rolling_std_1h'] = df['Consumption'].rolling('1h', closed='left').std()

# create lag features
df['lag_1'] = df['Consumption'].shift(1)
df['lag_2'] = df['Consumption'].shift(2)
df['lag_3'] = df['Consumption'].shift(3)

# Encode hour of day and day of week as cyclic features
df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day']/24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day']/24)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week']/7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week']/7)

df = df.reset_index() # reset the index
df.to_csv("energy_consumption_features.csv", index=False) # save to csv
print(df.head()) # preview the dataset
```

                 datetime  Unnamed: 0      TxnDate   TxnTime  Consumption  \
    0 2022-01-01 00:22:28       77476  01 Jan 2022  00:22:28        1.030   
    1 2022-01-01 00:42:33       76780  01 Jan 2022  00:42:33        0.904   
    2 2022-01-01 01:01:51       76954  01 Jan 2022  01:01:51        0.904   
    3 2022-01-01 01:41:48       76548  01 Jan 2022  01:41:48        1.850   
    4 2022-01-01 02:22:52       77070  01 Jan 2022  02:22:52        1.878   
    
       hour_of_day  day_of_week  is_weekend  rolling_mean_1h  rolling_std_1h  \
    0            0            5           1              NaN             NaN   
    1            0            5           1            1.030             NaN   
    2            1            5           1            0.967        0.089095   
    3            1            5           1            0.904        0.000000   
    4            2            5           1            1.850             NaN   
    
       lag_1  lag_2  lag_3  hour_sin  hour_cos   dow_sin   dow_cos  
    0    NaN    NaN    NaN  0.000000  1.000000 -0.974928 -0.222521  
    1  1.030    NaN    NaN  0.000000  1.000000 -0.974928 -0.222521  
    2  0.904  1.030    NaN  0.258819  0.965926 -0.974928 -0.222521  
    3  0.904  0.904  1.030  0.258819  0.965926 -0.974928 -0.222521  
    4  1.850  0.904  0.904  0.500000  0.866025 -0.974928 -0.222521  


### Exploratory Data Analysis (EDA)


```python
# Load the processed dataset
df = pd.read_csv("energy_consumption_features.csv", parse_dates=['datetime'])

# Figure: Daily total energy consumption trend
daily_consumption = df.groupby(df['datetime'].dt.date)['Consumption'].sum()
plt.figure(figsize=(10,5))
plt.plot(daily_consumption.index, daily_consumption.values, marker='o')
plt.title("Daily Total Energy Consumption (kWh)")
plt.xlabel("Date")
plt.ylabel("Total Consumption (kWh)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


    
![png](output_7_0.png)
    



```python
# Figure: Average energy consumption by hour of day
hourly_avg = df.groupby('hour_of_day')['Consumption'].mean()
plt.figure(figsize=(10,5))
plt.bar(hourly_avg.index, hourly_avg.values)
plt.title("Average Energy Consumption by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Consumption (kWh)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
```


    
![png](output_8_0.png)
    



```python
# Figure: Anomaly detection using 95th percentile as threshold
threshold = df['Consumption'].quantile(0.95)
df['is_anomaly'] = df['Consumption'] > threshold

plt.figure(figsize=(12,5))
plt.plot(df['datetime'], df['Consumption'], label='Consumption', color='blue')
plt.scatter(df.loc[df['is_anomaly'], 'datetime'],
            df.loc[df['is_anomaly'], 'Consumption'],
            color='red', label='Anomaly')
plt.title("Energy Consumption with Anomalies")
plt.xlabel("Datetime")
plt.ylabel("Consumption (kWh)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```


    
![png](output_9_0.png)
    


### Modelling


```python
# Define X, y
X = df[['rolling_mean_1h', 'rolling_std_1h', 'lag_1', 'lag_2', 'lag_3',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']]
y = df['Consumption']
```


```python
def smape_score(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)
    ) * 100

models = {
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=32, n_jobs=-1),
    "LightGBM": LGBMRegressor(n_estimators=1000, learning_rate=0.005, random_state=32, n_jobs=-1, 
                              min_child_samples=20, min_split_gain=0.0, max_depth=12, verbose=-1),
    "XGBoost": XGBRegressor(n_estimators=500, learning_rate=0.005, random_state=32, n_jobs=-1, tree_method="hist")
}

tscv = TimeSeriesSplit(n_splits=5)

results = {name: [] for name in models.keys()}

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # 去掉 NaN
    mask_train = ~X_train.isna().any(axis=1)
    X_train, y_train = X_train[mask_train], y_train[mask_train]
    mask_test = ~X_test.isna().any(axis=1)
    X_test, y_test = X_test[mask_test], y_test[mask_test]

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        smape_val = smape_score(y_test, y_pred)

        results[name].append((rmse, mae, r2, smape_val))
        print(f"{name} | Fold {fold+1}: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}, SMAPE={smape_val:.2f}%")

# Average results
print("\nAverage performance across folds:")
for name in models.keys():
    avg_rmse = np.mean([r[0] for r in results[name]])
    avg_mae  = np.mean([r[1] for r in results[name]])
    avg_r2   = np.mean([r[2] for r in results[name]])
    avg_smape = np.mean([r[3] for r in results[name]])
    print(f"{name}: RMSE={avg_rmse:.3f}, MAE={avg_mae:.3f}, R²={avg_r2:.3f}, SMAPE={avg_smape:.2f}%")
```

    RandomForest | Fold 1: RMSE=1.185, MAE=0.717, R²=0.817, SMAPE=29.15%
    LightGBM | Fold 1: RMSE=1.064, MAE=0.622, R²=0.852, SMAPE=26.76%
    XGBoost | Fold 1: RMSE=1.150, MAE=0.757, R²=0.827, SMAPE=33.32%
    RandomForest | Fold 2: RMSE=1.045, MAE=0.597, R²=0.861, SMAPE=17.77%
    LightGBM | Fold 2: RMSE=1.075, MAE=0.637, R²=0.853, SMAPE=21.05%
    XGBoost | Fold 2: RMSE=1.058, MAE=0.649, R²=0.857, SMAPE=22.46%
    RandomForest | Fold 3: RMSE=0.750, MAE=0.398, R²=0.748, SMAPE=21.39%
    LightGBM | Fold 3: RMSE=0.710, MAE=0.385, R²=0.775, SMAPE=21.46%
    XGBoost | Fold 3: RMSE=0.752, MAE=0.473, R²=0.747, SMAPE=28.73%
    RandomForest | Fold 4: RMSE=0.947, MAE=0.535, R²=0.779, SMAPE=21.98%
    LightGBM | Fold 4: RMSE=0.930, MAE=0.537, R²=0.787, SMAPE=21.97%
    XGBoost | Fold 4: RMSE=0.934, MAE=0.555, R²=0.785, SMAPE=24.71%
    RandomForest | Fold 5: RMSE=0.636, MAE=0.360, R²=0.852, SMAPE=18.69%
    LightGBM | Fold 5: RMSE=0.571, MAE=0.308, R²=0.881, SMAPE=15.63%
    XGBoost | Fold 5: RMSE=0.600, MAE=0.388, R²=0.869, SMAPE=22.65%
    
    Average performance across folds:
    RandomForest: RMSE=0.912, MAE=0.521, R²=0.812, SMAPE=21.80%
    LightGBM: RMSE=0.870, MAE=0.498, R²=0.830, SMAPE=21.37%
    XGBoost: RMSE=0.899, MAE=0.564, R²=0.817, SMAPE=26.37%



```python
features = ['Consumption', 'rolling_mean_1h', 'rolling_std_1h', 'lag_1', 'lag_2', 'lag_3',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']

mask = ~df[features].isna().any(axis=1) & ~df['Consumption'].isna()
data = df[features][mask].values
target = df['Consumption'][mask].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

timesteps = 50

X, y = [], []
for i in range(timesteps, len(data_scaled)):
    X.append(data_scaled[i-timesteps:i])
    y.append(target[i])

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)

```

    (2982, 50, 11) (2982,)



```python
model = Sequential()
model.add(Input(shape=(X.shape[1], X.shape[2])))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(1))


model.compile(optimizer=Adam(learning_rate=0.0025), loss='mse', metrics=['mae'])

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X, y,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    shuffle=False
)
```

    Epoch 1/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 9ms/step - loss: 6.7217 - mae: 1.6988 - val_loss: 1.8622 - val_mae: 1.0457
    Epoch 2/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 3.1470 - mae: 1.1517 - val_loss: 1.2837 - val_mae: 0.7283
    Epoch 3/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 2.3441 - mae: 0.9239 - val_loss: 0.9883 - val_mae: 0.6396
    Epoch 4/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 2.0454 - mae: 0.8534 - val_loss: 0.8404 - val_mae: 0.5693
    Epoch 5/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.7947 - mae: 0.7984 - val_loss: 0.7685 - val_mae: 0.5424
    Epoch 6/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - loss: 1.6748 - mae: 0.7553 - val_loss: 0.7009 - val_mae: 0.5161
    Epoch 7/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.5862 - mae: 0.7376 - val_loss: 0.6809 - val_mae: 0.5238
    Epoch 8/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.6010 - mae: 0.7323 - val_loss: 0.6412 - val_mae: 0.4879
    Epoch 9/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.4907 - mae: 0.7025 - val_loss: 0.6104 - val_mae: 0.4689
    Epoch 10/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.5586 - mae: 0.7208 - val_loss: 0.6061 - val_mae: 0.4774
    Epoch 11/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - loss: 1.5098 - mae: 0.7107 - val_loss: 0.5920 - val_mae: 0.4828
    Epoch 12/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.4656 - mae: 0.7047 - val_loss: 0.5889 - val_mae: 0.4801
    Epoch 13/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.4804 - mae: 0.7141 - val_loss: 0.5670 - val_mae: 0.4497
    Epoch 14/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.3822 - mae: 0.6796 - val_loss: 0.5686 - val_mae: 0.4406
    Epoch 15/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.3948 - mae: 0.6757 - val_loss: 0.5742 - val_mae: 0.4482
    Epoch 16/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.4121 - mae: 0.6832 - val_loss: 0.5611 - val_mae: 0.4227
    Epoch 17/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.3455 - mae: 0.6688 - val_loss: 0.5471 - val_mae: 0.4045
    Epoch 18/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.3653 - mae: 0.6695 - val_loss: 0.5705 - val_mae: 0.4401
    Epoch 19/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.3483 - mae: 0.6716 - val_loss: 0.5747 - val_mae: 0.4409
    Epoch 20/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.3216 - mae: 0.6592 - val_loss: 0.5449 - val_mae: 0.4233
    Epoch 21/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.3531 - mae: 0.6586 - val_loss: 0.5620 - val_mae: 0.4514
    Epoch 22/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.2955 - mae: 0.6428 - val_loss: 0.5337 - val_mae: 0.4191
    Epoch 23/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 11ms/step - loss: 1.3265 - mae: 0.6572 - val_loss: 0.5479 - val_mae: 0.4129
    Epoch 24/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.2942 - mae: 0.6455 - val_loss: 0.5297 - val_mae: 0.4160
    Epoch 25/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.3438 - mae: 0.6630 - val_loss: 0.5271 - val_mae: 0.4258
    Epoch 26/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.2879 - mae: 0.6406 - val_loss: 0.5312 - val_mae: 0.4084
    Epoch 27/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 10ms/step - loss: 1.3085 - mae: 0.6416 - val_loss: 0.5352 - val_mae: 0.4163
    Epoch 28/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.2769 - mae: 0.6395 - val_loss: 0.5295 - val_mae: 0.4107
    Epoch 29/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - loss: 1.2681 - mae: 0.6361 - val_loss: 0.5278 - val_mae: 0.4104
    Epoch 30/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.2904 - mae: 0.6382 - val_loss: 0.5282 - val_mae: 0.4021
    Epoch 31/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.2292 - mae: 0.6195 - val_loss: 0.5216 - val_mae: 0.3903
    Epoch 32/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.1977 - mae: 0.6123 - val_loss: 0.5407 - val_mae: 0.4099
    Epoch 33/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.1891 - mae: 0.6153 - val_loss: 0.5249 - val_mae: 0.4061
    Epoch 34/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.2273 - mae: 0.6268 - val_loss: 0.5262 - val_mae: 0.4142
    Epoch 35/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.2353 - mae: 0.6224 - val_loss: 0.5333 - val_mae: 0.4032
    Epoch 36/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.1937 - mae: 0.6069 - val_loss: 0.5246 - val_mae: 0.3973
    Epoch 37/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 9ms/step - loss: 1.1852 - mae: 0.6137 - val_loss: 0.5406 - val_mae: 0.4146
    Epoch 38/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.1899 - mae: 0.6126 - val_loss: 0.5512 - val_mae: 0.4204
    Epoch 39/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.1724 - mae: 0.6035 - val_loss: 0.5358 - val_mae: 0.4189
    Epoch 40/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 7ms/step - loss: 1.1827 - mae: 0.6037 - val_loss: 0.5367 - val_mae: 0.4042
    Epoch 41/100
    [1m75/75[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 8ms/step - loss: 1.2150 - mae: 0.6073 - val_loss: 0.5674 - val_mae: 0.4119



```python
y_pred = model.predict(X)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

rmse_lstm = np.sqrt(mean_squared_error(y, y_pred))
mae_lstm = mean_absolute_error(y, y_pred)
r2_lstm = r2_score(y, y_pred)
smape_lstm = smape_score(y, y_pred)

print(f"LSTM result: RMSE={rmse_lstm:.3f}, MAE={mae_lstm:.3f}, R²={r2_lstm:.3f}, SMAPE={smape_lstm:.2f}%")

```

    [1m94/94[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 2ms/step
    LSTM result: RMSE=0.993, MAE=0.501, R²=0.840, SMAPE=72.32%



```python
# Print the summary
summary = []
for name in results.keys():
    avg_rmse = np.mean([r[0] for r in results[name]])
    avg_mae  = np.mean([r[1] for r in results[name]])
    avg_r2   = np.mean([r[2] for r in results[name]])
    avg_smape = np.mean([r[3] for r in results[name]])
    summary.append([name, avg_rmse, avg_mae, avg_r2, avg_smape])

summary.append(["LSTM", rmse_lstm, mae_lstm, r2_lstm, smape_lstm])

df_results = pd.DataFrame(summary, columns=["Model", "RMSE", "MAE", "R²", "SMAPE (%)"])
df_results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>RMSE</th>
      <th>MAE</th>
      <th>R²</th>
      <th>SMAPE (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>RandomForest</td>
      <td>0.912416</td>
      <td>0.521408</td>
      <td>0.811542</td>
      <td>21.798056</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LightGBM</td>
      <td>0.869596</td>
      <td>0.497750</td>
      <td>0.829625</td>
      <td>21.372087</td>
    </tr>
    <tr>
      <th>2</th>
      <td>XGBoost</td>
      <td>0.898574</td>
      <td>0.564431</td>
      <td>0.817175</td>
      <td>26.374506</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LSTM</td>
      <td>0.993098</td>
      <td>0.501352</td>
      <td>0.840049</td>
      <td>72.322498</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python
# 
hourly = (
    df.set_index('datetime')
      .resample('1h')
      .agg(Consumption=('Consumption','mean'), obs_per_hour=('Consumption','size'))
)
hourly
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Consumption</th>
      <th>obs_per_hour</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-01-01 00:00:00</th>
      <td>0.967000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2022-01-01 01:00:00</th>
      <td>1.377000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2022-01-01 02:00:00</th>
      <td>1.485000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2022-01-01 03:00:00</th>
      <td>0.929333</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2022-01-01 04:00:00</th>
      <td>0.922667</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2022-02-28 19:00:00</th>
      <td>0.968000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2022-02-28 20:00:00</th>
      <td>0.932000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2022-02-28 21:00:00</th>
      <td>0.960000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2022-02-28 22:00:00</th>
      <td>1.020000</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2022-02-28 23:00:00</th>
      <td>1.033333</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>1416 rows × 2 columns</p>
</div>




```python

# 2) 缺值處理：先用線性插值補 Consumption，再用 0 填 obs_per_hour（沒有資料的時段）
hourly['Consumption'] = hourly['Consumption'].interpolate(limit_direction='both')
hourly['obs_per_hour'] = hourly['obs_per_hour'].fillna(0)

# 3) 時間特徵
hourly['hour_of_day'] = hourly.index.hour
hourly['day_of_week'] = hourly.index.dayofweek
hourly['is_weekend']  = (hourly['day_of_week'] >= 5).astype(int)

# create lag features
hourly['lag_1'] = hourly['Consumption'].shift(1)
hourly['lag_2'] = hourly['Consumption'].shift(2)
hourly['lag_3'] = hourly['Consumption'].shift(3)
#hourly['lag_24'] = hourly['Consumption'].shift(24)

# Encode hour of day and day of week as cyclic features
hourly['hour_sin'] = np.sin(2 * np.pi * hourly['hour_of_day']/24)
hourly['hour_cos'] = np.cos(2 * np.pi * hourly['hour_of_day']/24)
hourly['dow_sin'] = np.sin(2 * np.pi * hourly['day_of_week']/7)
hourly['dow_cos'] = np.cos(2 * np.pi * hourly['day_of_week']/7)

# 5) 丟掉因滯後產生的 NaN 開頭
hourly = hourly.dropna()

# 6) 切分訓練/驗證（時間序）
split_idx = int(len(hourly)*0.8)
train_df = hourly.iloc[:split_idx]
val_df   = hourly.iloc[split_idx:]

# 7) 標準化（只在訓練集 fit）
from sklearn.preprocessing import StandardScaler
features = ['Consumption', 'is_weekend', 'lag_1', 'lag_2', 'lag_3', #'lag_24',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(train_df[features])
X_val_scaled   = scaler.transform(val_df[features])

y_train = train_df['Consumption'].values
y_val   = val_df['Consumption'].values

# 8) 做 LSTM 序列（例如用過去 168 小時預測下一小時）
def make_sequences(X2d, y1d, lookback=168):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X2d)):
        X_seq.append(X2d[i-lookback:i, :])
        y_seq.append(y1d[i])
    return np.array(X_seq), np.array(y_seq)

lookback = 168
Xtr, ytr = make_sequences(X_train_scaled, y_train, lookback)
Xva, yva = make_sequences(X_val_scaled,   y_val,   lookback)

# 9) 建 LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

model = Sequential([
    Input(shape=(Xtr.shape[1], Xtr.shape[2])),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    Xtr, ytr,
    epochs=100,
    batch_size=32,
    validation_data=(Xva, yva),
    callbacks=[es],
    shuffle=False
)

```

    Epoch 1/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 27ms/step - loss: 14.5087 - mae: 2.5223 - val_loss: 1.3814 - val_mae: 0.7360
    Epoch 2/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 9.2531 - mae: 2.1517 - val_loss: 1.1586 - val_mae: 0.9038
    Epoch 3/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 7.2494 - mae: 1.8473 - val_loss: 1.1975 - val_mae: 0.9229
    Epoch 4/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 6.6025 - mae: 1.8052 - val_loss: 1.1176 - val_mae: 0.7829
    Epoch 5/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 6.2803 - mae: 1.7040 - val_loss: 1.0766 - val_mae: 0.7708
    Epoch 6/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 29ms/step - loss: 5.9792 - mae: 1.6411 - val_loss: 1.0677 - val_mae: 0.7580
    Epoch 7/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 5.7823 - mae: 1.6022 - val_loss: 1.0653 - val_mae: 0.7310
    Epoch 8/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 5.4631 - mae: 1.5458 - val_loss: 1.1003 - val_mae: 0.7534
    Epoch 9/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 5.2670 - mae: 1.4880 - val_loss: 0.9909 - val_mae: 0.7362
    Epoch 10/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 4.8953 - mae: 1.4223 - val_loss: 0.9453 - val_mae: 0.7138
    Epoch 11/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 4.6913 - mae: 1.3885 - val_loss: 0.8874 - val_mae: 0.7169
    Epoch 12/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 4.4514 - mae: 1.3396 - val_loss: 0.9084 - val_mae: 0.7856
    Epoch 13/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 4.2889 - mae: 1.3232 - val_loss: 0.8211 - val_mae: 0.6626
    Epoch 14/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 4.2123 - mae: 1.3026 - val_loss: 0.8718 - val_mae: 0.7388
    Epoch 15/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 4.1342 - mae: 1.2597 - val_loss: 0.8075 - val_mae: 0.7177
    Epoch 16/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 4.0174 - mae: 1.2462 - val_loss: 0.8094 - val_mae: 0.7055
    Epoch 17/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 3.6509 - mae: 1.2046 - val_loss: 0.8938 - val_mae: 0.7176
    Epoch 18/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 3.6601 - mae: 1.1659 - val_loss: 0.8972 - val_mae: 0.7336
    Epoch 19/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 3.5768 - mae: 1.1732 - val_loss: 0.8552 - val_mae: 0.6874
    Epoch 20/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 3.3826 - mae: 1.1446 - val_loss: 0.9113 - val_mae: 0.7084
    Epoch 21/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 3.4980 - mae: 1.1563 - val_loss: 0.8689 - val_mae: 0.6766
    Epoch 22/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 3.2641 - mae: 1.1398 - val_loss: 0.9227 - val_mae: 0.7103
    Epoch 23/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 3.2931 - mae: 1.1294 - val_loss: 0.7373 - val_mae: 0.6497
    Epoch 24/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 3.2963 - mae: 1.1234 - val_loss: 0.6300 - val_mae: 0.6410
    Epoch 25/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 3.1545 - mae: 1.1189 - val_loss: 0.5714 - val_mae: 0.6148
    Epoch 26/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 3.2201 - mae: 1.1217 - val_loss: 1.6574 - val_mae: 0.9112
    Epoch 27/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 3.1667 - mae: 1.1278 - val_loss: 0.6601 - val_mae: 0.6222
    Epoch 28/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 33ms/step - loss: 3.0894 - mae: 1.1109 - val_loss: 0.6157 - val_mae: 0.5937
    Epoch 29/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.9730 - mae: 1.0835 - val_loss: 0.6570 - val_mae: 0.6125
    Epoch 30/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 2.9008 - mae: 1.0785 - val_loss: 0.5898 - val_mae: 0.5917
    Epoch 31/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 2.9441 - mae: 1.0377 - val_loss: 0.7599 - val_mae: 0.6298
    Epoch 32/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 2.8025 - mae: 1.0541 - val_loss: 0.9023 - val_mae: 0.6973
    Epoch 33/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.8120 - mae: 1.0650 - val_loss: 0.5449 - val_mae: 0.5587
    Epoch 34/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.8554 - mae: 1.0482 - val_loss: 0.5784 - val_mae: 0.5764
    Epoch 35/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.6749 - mae: 1.0299 - val_loss: 0.5235 - val_mae: 0.5540
    Epoch 36/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.7858 - mae: 1.0470 - val_loss: 0.5071 - val_mae: 0.5269
    Epoch 37/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 22ms/step - loss: 2.5455 - mae: 0.9974 - val_loss: 0.5001 - val_mae: 0.5368
    Epoch 38/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.5916 - mae: 0.9884 - val_loss: 0.5851 - val_mae: 0.5702
    Epoch 39/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.7101 - mae: 1.0221 - val_loss: 0.8638 - val_mae: 0.6714
    Epoch 40/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 2.7473 - mae: 1.0149 - val_loss: 0.5868 - val_mae: 0.5600
    Epoch 41/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.6210 - mae: 0.9964 - val_loss: 0.5211 - val_mae: 0.5434
    Epoch 42/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 2.5966 - mae: 1.0018 - val_loss: 1.0134 - val_mae: 0.7071
    Epoch 43/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 2.8198 - mae: 1.0418 - val_loss: 0.5048 - val_mae: 0.5278
    Epoch 44/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.6022 - mae: 1.0087 - val_loss: 0.6235 - val_mae: 0.5795
    Epoch 45/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.4248 - mae: 0.9795 - val_loss: 0.5203 - val_mae: 0.5429
    Epoch 46/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.4137 - mae: 0.9769 - val_loss: 0.4960 - val_mae: 0.5423
    Epoch 47/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.5090 - mae: 0.9573 - val_loss: 0.6188 - val_mae: 0.5843
    Epoch 48/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 2.4271 - mae: 0.9730 - val_loss: 0.5085 - val_mae: 0.5447
    Epoch 49/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 2.3129 - mae: 0.9377 - val_loss: 0.5280 - val_mae: 0.5571
    Epoch 50/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.2890 - mae: 0.9238 - val_loss: 0.6305 - val_mae: 0.5834
    Epoch 51/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.3403 - mae: 0.9495 - val_loss: 0.4901 - val_mae: 0.5036
    Epoch 52/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.2941 - mae: 0.9115 - val_loss: 0.7530 - val_mae: 0.6338
    Epoch 53/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.1654 - mae: 0.9058 - val_loss: 0.6789 - val_mae: 0.6018
    Epoch 54/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 2.3435 - mae: 0.9413 - val_loss: 0.9394 - val_mae: 0.6849
    Epoch 55/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.4215 - mae: 0.9958 - val_loss: 0.4529 - val_mae: 0.4833
    Epoch 56/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.3336 - mae: 0.9897 - val_loss: 0.7120 - val_mae: 0.6112
    Epoch 57/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.2820 - mae: 0.9242 - val_loss: 0.9209 - val_mae: 0.7308
    Epoch 58/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 23ms/step - loss: 2.0381 - mae: 0.9059 - val_loss: 0.8712 - val_mae: 0.6048
    Epoch 59/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 24ms/step - loss: 2.2566 - mae: 0.9244 - val_loss: 0.9261 - val_mae: 0.6931
    Epoch 60/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 29ms/step - loss: 2.2122 - mae: 0.9301 - val_loss: 0.6596 - val_mae: 0.5725
    Epoch 61/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 2.0507 - mae: 0.9170 - val_loss: 0.8195 - val_mae: 0.6803
    Epoch 62/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m2s[0m 58ms/step - loss: 2.2232 - mae: 0.9360 - val_loss: 0.5305 - val_mae: 0.5197
    Epoch 63/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 38ms/step - loss: 2.0939 - mae: 0.8732 - val_loss: 1.0757 - val_mae: 0.7250
    Epoch 64/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 2.3368 - mae: 0.9389 - val_loss: 1.2742 - val_mae: 0.7218
    Epoch 65/100
    [1m30/30[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m1s[0m 25ms/step - loss: 2.2498 - mae: 0.9256 - val_loss: 0.6570 - val_mae: 0.5740



```python
# Predict on validation set
y_pred = model.predict(Xva).flatten()

# Compare with true values
print("y_val shape:", y_val.shape, "y_pred shape:", y_pred.shape)

```

    [1m4/4[0m [32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m0s[0m 35ms/step
    y_val shape: (276,) y_pred shape: (108,)



```python
rmse = np.sqrt(mean_squared_error(y_val[lookback:], y_pred))
mae = mean_absolute_error(y_val[lookback:], y_pred)
r2 = r2_score(y_val[lookback:], y_pred)

print(f"LSTM on validation: RMSE={rmse:.3f}, MAE={mae:.3f}, R²={r2:.3f}")

```

    LSTM on validation: RMSE=0.673, MAE=0.483, R²=0.430



```python

```
