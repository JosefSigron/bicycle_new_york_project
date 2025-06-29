# CitiBike Ride Prediction System - Complete Guide

## 🚀 Overview

This system predicts CitiBike ridership patterns using machine learning and deep learning. It transforms millions of individual ride records into predictable hourly and daily forecasts for business planning.

---

## 🎯 What We Predict: The 4 Targets

### **1. Hourly Ride Count** 
- **What**: Number of rides that start in each hour
- **Example**: "At 8 AM on Tuesday, there will be ~2,500 rides"
- **Use Case**: Short-term bike availability planning

### **2. Hourly Average Duration**
- **What**: Average length (in seconds) of rides starting in each hour  
- **Example**: "Rides starting at 8 AM will average ~600 seconds (10 minutes)"
- **Use Case**: Fleet turnover planning, station capacity

### **3. Daily Ride Count**
- **What**: Total number of rides in each day
- **Example**: "Monday will have ~45,000 total rides"
- **Use Case**: Strategic planning, maintenance scheduling

### **4. Daily Average Duration**
- **What**: Average length of all rides in each day
- **Example**: "Monday rides will average ~800 seconds (13 minutes)"
- **Use Case**: Usage pattern analysis, pricing strategies

### **Why These 4?**

**Time Granularity:**
- **Hourly** = Operational decisions (real-time management)
- **Daily** = Strategic decisions (long-term planning)

**Prediction Type:**
- **Count** = How many rides? (demand forecasting)
- **Duration** = How long per ride? (resource planning)

### **Business Applications:**

| Prediction | Business Use |
|------------|-------------|
| **Hourly Count** | "Deploy 20 extra bikes to Times Square at 8 AM" |
| **Hourly Duration** | "Expect bikes back in Central Park in ~15 minutes" |
| **Daily Count** | "Order maintenance for 50,000 rides tomorrow" |  
| **Daily Duration** | "Average usage will be light/heavy today" |

---

## 🔄 How Preprocessing Works

### **The Challenge**
- **Raw Data**: Millions of individual ride records (6+ GB)
- **Goal**: Transform into predictable patterns for machine learning

### **Step 1: Raw Data Structure**
```
Each row = 1 individual ride
Columns: start_time, stop_time, trip_duration, weather, location, etc.

Example rows:
2024-01-15 08:15:23 | 2024-01-15 08:28:41 | 798 sec | 15°C | rain=0 | Manhattan
2024-01-15 08:16:45 | 2024-01-15 08:31:22 | 877 sec | 15°C | rain=0 | Brooklyn  
2024-01-15 08:17:12 | 2024-01-15 08:25:33 | 501 sec | 15°C | rain=0 | Manhattan
...millions of rows...
```

### **Step 2: Memory-Efficient Loading**
```python
# Instead of loading all 6GB at once:
for chunk in read_parquet(file, chunksize=100000):
    process_chunk(chunk)
    combine_results()
    garbage_collect()  # Free memory
```

**Loading Options:**
- **Lightweight**: 500K rows (~200MB, 5-10 min)
- **Single Year Sample**: 10% of 2024 (~400MB, 10-15 min)
- **Single Year Full**: All 2024 (~1.5GB, 30-45 min)
- **Recent Years Sample**: 10% of 2022-2024 (~800MB, 20-30 min)
- **Full Dataset**: All years (~6GB+, 2+ hours)

### **Step 3: Time Feature Engineering**
```python
# From timestamp, extract multiple time features:
df['start_time'] = '2024-01-15 08:15:23'

# Becomes:
df['hour'] = 8
df['day_of_week'] = 1  # Monday=0, Tuesday=1, etc.
df['month'] = 1
df['season'] = 'Winter'
df['is_weekend'] = 0
df['time_period'] = 'Morning'  # Night/Morning/Afternoon/Evening
df['quarter'] = 1
df['day_of_year'] = 15
```

### **Step 4: Hourly Aggregation**
```python
# Group individual rides by hour and aggregate:

Original data (same hour):
2024-01-15 08:15 | 798 sec | 15°C | rain=0
2024-01-15 08:16 | 877 sec | 15°C | rain=0  
2024-01-15 08:17 | 501 sec | 15°C | rain=0
...1,247 more rides in this hour...

Becomes aggregated hour:
hour_timestamp: 2024-01-15 08:00
ride_count: 1,250          # Count of rides
avg_duration: 650          # Average of all durations
median_duration: 600       # Median duration
std_duration: 180          # Standard deviation
min_duration: 301          # Shortest ride
max_duration: 1,205        # Longest ride
temperature: 15            # Average temperature
wind_speed: 12            # Average wind speed
precipitation: 0          # Total precipitation
rain: 0                   # Most common rain value
weather_cat: 'Clear'      # Most common weather
hour: 8                   # Time features
day_of_week: 1
is_weekend: 0
user_type: 0.85           # Proportion of subscribers
```

### **Step 5: Creating Historical Features**
```python
# Add lag features (previous hours' performance):
ride_count_lag_1h: 1,180   # Rides 1 hour ago
ride_count_lag_2h: 950     # Rides 2 hours ago
ride_count_lag_6h: 890     # Rides 6 hours ago
ride_count_lag_12h: 780    # Rides 12 hours ago
ride_count_lag_24h: 1,240  # Rides same time yesterday

# Add rolling averages (trends):
ride_count_rolling_3h: 1,150   # Average of last 3 hours
ride_count_rolling_6h: 1,080   # Average of last 6 hours
ride_count_rolling_12h: 980    # Average of last 12 hours
ride_count_rolling_24h: 990    # Average of last 24 hours

# Same for duration:
avg_duration_lag_1h: 580
avg_duration_rolling_3h: 615
```

### **Step 6: Daily Aggregation (Parallel Process)**
```python
# Similar process but group by day:

All rides on 2024-01-15:
08:00 hour: 1,250 rides, avg 650 sec
09:00 hour: 2,100 rides, avg 580 sec
10:00 hour: 1,800 rides, avg 720 sec
...24 hours of data...

Becomes daily summary:
date: 2024-01-15
daily_ride_count: 45,230        # Total rides that day
daily_avg_duration: 620         # Average duration all day
daily_median_duration: 580      # Median duration
daily_std_duration: 180         # Standard deviation
temperature_mean: 12            # Daily average temperature  
temperature_min: 8              # Daily minimum temperature
temperature_max: 18             # Daily maximum temperature
precipitation_sum: 2.3          # Total rainfall that day
rain: 1                         # Any rain during the day
user_type: 0.82                 # Daily subscriber ratio

# Plus lag features:
daily_ride_count_lag_1d: 43,500   # Yesterday's rides
daily_ride_count_lag_7d: 38,200   # Same day last week
daily_ride_count_rolling_7d: 41,000  # Average last 7 days
```

### **Processing Flow Diagram**

```
Raw Individual Rides (millions of rows)
         ↓
📅 Add Time Features (hour, day, season, etc.)
         ↓
🕐 Group by Hour → Hourly Statistics (thousands of rows)
         ↓  
📈 Add Lag Features (previous hours)
         ↓
📊 Add Rolling Averages (trends)
         ↓
🔧 Handle Categorical Data (weather, seasons)
         ↓
💾 Save: hourly_aggregated.parquet

         ↓ (parallel process)
         
📅 Group by Day → Daily Statistics (hundreds of rows)
         ↓
📈 Add Lag Features (previous days)
         ↓
📊 Add Rolling Averages (trends)
         ↓
🔧 Handle Categorical Data
         ↓
💾 Save: daily_aggregated.parquet
```

### **Feature Categories Created:**
- **Target Variables**: ride_count, avg_duration (what we predict)
- **Time Features**: hour, day_of_week, season, weekend, time_period
- **Weather Features**: temperature, precipitation, wind_speed, humidity, UTCI
- **Historical Features**: lag_1h, lag_24h, rolling_3h, rolling_24h
- **User Features**: subscriber_ratio
- **Location Features**: borough information

---

## 🤖 Machine Learning Models

We use **8 different models** across two categories to automatically find the best predictor for each task.

### **Traditional Machine Learning Models (6 models)**

#### **1. Linear Regression**
```python
LinearRegression()
```
- **How it works**: Finds straight-line relationships between features and target
- **Strengths**: Simple, fast, interpretable
- **Best for**: Clean patterns with linear relationships

#### **2. Ridge Regression** 
```python
Ridge(alpha=1.0)
```
- **How it works**: Linear regression with penalty for large coefficients
- **Strengths**: Prevents overfitting, handles many features
- **Best for**: When you have many correlated features

#### **3. Lasso Regression**
```python
Lasso(alpha=1.0) 
```
- **How it works**: Linear regression that can eliminate unimportant features
- **Strengths**: Automatic feature selection, sparse models
- **Best for**: Feature selection, reducing model complexity

#### **4. Random Forest** 🌳
```python
RandomForestRegressor(n_estimators=100, random_state=42)
```
- **How it works**: Combines 100 decision trees, averages their predictions
- **Strengths**: Handles non-linear patterns, provides feature importance
- **Best for**: Complex patterns, robust predictions

#### **5. Gradient Boosting** 🚀
```python
GradientBoostingRegressor(n_estimators=100, random_state=42)
```
- **How it works**: Builds trees sequentially, each correcting previous errors
- **Strengths**: High accuracy, handles complex patterns
- **Best for**: Maximum predictive performance

#### **6. Support Vector Regression (SVR)**
```python
SVR(kernel='rbf', C=1.0)
```
- **How it works**: Finds optimal boundary in high-dimensional space
- **Strengths**: Handles non-linear patterns, robust to outliers
- **Best for**: Non-linear relationships, noisy data

### **Deep Learning Models (2 models)**

#### **7. Neural Network (Feedforward)**
```python
Sequential([
    Dense(128, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1)
])
```
- **How it works**: Multi-layer network learns complex non-linear patterns
- **Architecture**: 128→64→32→1 neurons with dropout for regularization
- **Strengths**: Can learn any pattern, handles complex interactions
- **Best for**: Large datasets, complex non-linear relationships

#### **8. LSTM (Long Short-Term Memory)**
```python
Sequential([
    LSTM(50, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2), 
    Dense(25),
    Dense(1)
])
```
- **How it works**: Processes sequences of 24 hours to predict next hour
- **Architecture**: Takes last 24 hours of data → predicts next hour
- **Strengths**: Designed for time series, learns temporal dependencies
- **Best for**: Sequential patterns, time series forecasting

### **Model Selection Strategy**

```python
# Automatic model comparison process:
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
    'SVR': SVR(kernel='rbf', C=1.0),
    'Neural Network': create_neural_network(),
    'LSTM': create_lstm_model()
}

# Train all models
for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    performance = evaluate(y_test, predictions)
    
# Automatically select best performer
best_model = select_best_model(models, performance_metrics)
```

### **Performance Metrics:**
- **RMSE**: Root Mean Square Error (lower = better)
- **MAE**: Mean Absolute Error (lower = better) 
- **R²**: R-squared (higher = better, 1.0 = perfect, negative = worse than mean)

The system automatically compares all models and selects the best performer for each prediction task based on cross-validation results.

### **Model Characteristics:**

#### **Simple Models (Linear, Ridge, Lasso):**
- ✅ Fast training and prediction
- ✅ Highly interpretable results
- ✅ Stable across different datasets
- ✅ Work well with clean time/weather patterns
- ❌ Can't capture complex interactions
- ❌ Assume linear relationships

#### **Tree Models (Random Forest, Gradient Boosting):**
- ✅ Handle non-linear patterns automatically
- ✅ Provide feature importance rankings
- ✅ Robust to outliers and missing data
- ✅ No assumption about data distribution
- ❌ Can overfit with small datasets
- ❌ Less interpretable than linear models

#### **Deep Learning (Neural Networks, LSTM):**
- ✅ Learn arbitrarily complex patterns
- ✅ Can handle large feature sets
- ✅ Automatic feature interaction discovery
- ❌ Need more data to perform well
- ❌ Less interpretable ("black box")
- ❌ Prone to overfitting with small datasets
- ❌ Computationally expensive

#### **Time Series Specific (LSTM):**
- ✅ Designed for sequential data
- ✅ Learns temporal dependencies automatically
- ✅ Can capture long-term patterns
- ❌ Needs lots of historical data
- ❌ Computationally expensive
- ❌ Complex to tune properly

---

## 🚀 Quick Start

### **Basic Usage (Recommended)**
```bash
# Fast testing with lightweight data
python src/run_ride_prediction.py
```

### **Custom Data Amount**
```bash
# Interactive menu with data size options
python src/run_full_prediction.py
```

### **Memory Requirements**
- **Lightweight**: 4GB RAM, 5-10 minutes
- **Full dataset**: 8GB+ RAM, 1-2 hours

---

## 📊 Results Interpretation

### **Performance Indicators:**
- **R² > 0.5**: Excellent predictive power
- **R² 0.3-0.5**: Good predictive power  
- **R² 0.1-0.3**: Moderate predictive power
- **R² < 0.1**: Poor predictive power
- **R² < 0**: Model worse than simply predicting the mean

### **Prediction Challenges:**

**Ride Count Predictions:**
- Generally more predictable due to clear temporal patterns
- Weather and time of day are strong predictors
- Aggregated patterns are more stable than individual behavior

**Duration Predictions:**
- More challenging due to individual variation in riding behavior
- Affected by factors not captured in the data (traffic, route choice, stops)
- Extreme outliers in the data (measurement errors, unusually long rides)

---

## 🎯 Business Value

### **Operational Planning:**
- **Fleet Management**: Predict bike availability needs
- **Maintenance Scheduling**: Plan based on usage patterns
- **Staff Allocation**: Deploy resources to high-demand areas

### **Strategic Insights:**
- **Weather Impact**: Quantify weather effects on ridership
- **Seasonal Planning**: Prepare for demand variations
- **Growth Forecasting**: Project future system needs

### **Real-Time Applications:**
- **Dynamic Pricing**: Adjust pricing based on predicted demand
- **User Notifications**: Alert users about availability
- **Route Optimization**: Suggest optimal trip timing

---

## 📁 File Structure

```
bicycle_new_york_project/
├── CitiBike_Prediction_System_Guide.md     # This guide
├── src/
│   ├── run_ride_prediction.py              # Quick start script
│   ├── run_full_prediction.py              # Full dataset options
│   ├── ride_prediction_preprocessing.py    # Data preprocessing
│   ├── ride_prediction_models.py           # ML models
│   └── prediction_example.py               # Usage examples
├── data/
│   ├── combined/                           # Raw parquet files
│   └── processed/                          # Preprocessed data
├── results/                                # Model results and summaries
├── info/                                   # Detailed documentation
│   ├── ride_prediction_guide.md           # Technical guide
│   └── memory_management_guide.md         # Memory optimization
└── requirements.txt                        # Python dependencies
```

---

## 🔧 Next Steps for Improvement

### **For Better Duration Prediction:**
1. **Data Cleaning**: Remove extreme outliers (rides > 4 hours)
2. **More Data**: Use full dataset instead of samples
3. **Feature Engineering**: Add route distance, elevation, traffic data
4. **Advanced Models**: Try XGBoost, CatBoost, or Prophet

### **For Better Overall Performance:**
1. **External Data**: Add holidays, events, subway outages
2. **Spatial Features**: Station-level predictions
3. **Real-time Features**: Current weather conditions
4. **Ensemble Methods**: Combine multiple models

### **For Production Deployment:**
1. **Model Monitoring**: Track prediction accuracy over time
2. **Automated Retraining**: Update models with new data
3. **API Development**: Serve predictions via REST API
4. **Dashboard Creation**: Visualize predictions and performance

---

## 💡 Key Insights

1. **Aggregation is powerful**: Individual rides are unpredictable, but hourly/daily patterns can be modeled
2. **Time patterns are crucial**: Hour of day and day of week are fundamental features for prediction
3. **Weather matters**: Temperature and precipitation significantly influence ridership patterns
4. **Model diversity helps**: Different models excel at different prediction tasks
5. **Data quality is key**: Clean, consistent data improves all model performance

This system transforms the complex problem of predicting individual ride behavior into the manageable task of forecasting aggregate patterns - making it useful for real business decisions! 🚀 