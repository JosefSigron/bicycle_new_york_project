# CitiBike Ride Prediction System

## Overview

This system predicts CitiBike ride counts and durations using machine learning and deep learning techniques. It transforms individual ride data into aggregated time series and applies various prediction models.

## Problem Definition

**Challenge**: The raw data contains individual ride records, but we want to predict:
1. **Number of rides** (per hour/day)
2. **Average ride duration** (per hour/day)

**Solution**: Aggregate individual rides into time windows and create features for prediction.

## Data Pipeline

### 1. Data Structure
- **Input**: Parquet files with 37 columns per ride
- **Key Columns**: 
  - Time: `start_time`, `stop_time`, `trip_duration`
  - Location: Station IDs, coordinates, boroughs
  - Weather: `temperature`, `wind_speed`, `precipitation`, `utci`, etc.
  - User: `user_type` (Subscriber/Customer)

### 2. Preprocessing Steps

#### Time Feature Engineering
- Extract: hour, day of week, month, season, weekend indicator
- Create time periods: Night, Morning, Afternoon, Evening
- Add temporal lags (1h, 2h, 6h, 12h, 24h for hourly data)
- Add rolling averages (3h, 6h, 12h, 24h windows)

#### Aggregation Strategies

**Hourly Aggregation:**
- Count rides per hour → `ride_count`
- Average duration per hour → `avg_duration`
- Weather conditions (hourly averages)
- User type distribution

**Daily Aggregation:**
- Count rides per day → `daily_ride_count`
- Average duration per day → `daily_avg_duration`
- Daily weather summaries (min/max/avg temperature, total precipitation)

#### Feature Engineering
- **Lag Features**: Previous hours/days performance
- **Rolling Statistics**: Moving averages to capture trends
- **Weather Categorization**: Rain, snow, mist/fog indicators
- **Seasonal Patterns**: Monthly and seasonal effects
- **User Behavior**: Subscriber vs. customer ratios

## Machine Learning Models

### Traditional Models
1. **Linear Regression**: Baseline model
2. **Ridge/Lasso Regression**: Regularized linear models
3. **Random Forest**: Ensemble tree method (good for feature importance)
4. **Gradient Boosting**: Advanced ensemble method
5. **Support Vector Regression**: Non-linear patterns

### Deep Learning Models
1. **Neural Network**: Feedforward network with dropout
2. **LSTM**: Long Short-Term Memory for time series patterns

### Model Selection Strategy
- **Time Series Split**: Chronological train/test split (80/20)
- **Cross-Validation**: Time series cross-validation
- **Metrics**: RMSE, MAE, R²
- **Feature Importance**: Available for tree-based models

## Usage

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline
python src/run_ride_prediction.py
```

### Step-by-Step
```python
# 1. Preprocessing only
python src/ride_prediction_preprocessing.py

# 2. Models only (if data already processed)
python src/ride_prediction_models.py
```

### Custom Predictions
```python
from ride_prediction_preprocessing import RideDataPreprocessor
from ride_prediction_models import RidePredictionModels

# Load and preprocess data
preprocessor = RideDataPreprocessor()
data = preprocessor.load_data([2023, 2024])
hourly_data = preprocessor.create_hourly_aggregated_data()

# Train models
models = RidePredictionModels()
X, y, _ = preprocessor.get_feature_target_split('hourly', 'ride_count')
# ... training code
```

## Results Interpretation

### Performance Metrics
- **RMSE (Root Mean Square Error)**: Average prediction error in original units
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **R² (R-squared)**: Proportion of variance explained (0-1, higher is better)

### Typical Performance Expectations
- **Hourly Ride Count**: R² ≈ 0.7-0.9 (highly predictable)
- **Hourly Duration**: R² ≈ 0.3-0.6 (more variable)
- **Daily Aggregates**: Usually higher R² due to smoothing

### Feature Importance Insights
Key factors typically include:
1. **Time patterns**: Hour of day, day of week, season
2. **Weather**: Temperature, precipitation, wind
3. **Historical patterns**: Recent ride counts, trends
4. **User behavior**: Subscriber ratios

## Output Files

### Processed Data
- `data/processed/hourly_aggregated.parquet`: Hourly ride statistics
- `data/processed/daily_aggregated.parquet`: Daily ride statistics

### Results
- `results/prediction_summary.csv`: Model performance comparison
- Various plots: Feature importance, predictions vs. actual

### Example Code
- `src/prediction_example.py`: Template for making new predictions

## Advanced Features

### Time Series Considerations
- **Seasonality**: Models capture seasonal patterns
- **Trends**: Long-term changes in ridership
- **External Events**: Weather impacts, holidays (could be added)

### Model Ensemble
- Combine multiple models for better predictions
- Weight models based on performance
- Reduce overfitting through averaging

### Future Enhancements
1. **External Data**: Holidays, events, subway outages
2. **Spatial Models**: Station-level predictions
3. **Real-time Features**: Current weather conditions
4. **Advanced Architectures**: Transformer models, Prophet

## Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce years of data loaded
2. **Missing Dependencies**: Install TensorFlow for deep learning
3. **Data Format**: Ensure parquet files are in `data/combined/`

### Performance Optimization
1. **Feature Selection**: Remove less important features
2. **Model Selection**: Start with faster models (Random Forest)
3. **Data Sampling**: Use subset for quick testing

### Validation
1. **Visual Inspection**: Check prediction plots
2. **Residual Analysis**: Look for patterns in errors
3. **Cross-Validation**: Ensure model generalizes

## Business Applications

### Operational Planning
- **Fleet Management**: Predict bike availability needs
- **Maintenance Scheduling**: Plan based on usage patterns
- **Staff Allocation**: Deploy resources to high-demand areas

### Strategic Insights
- **Weather Impact**: Quantify weather effects on ridership
- **Seasonal Planning**: Prepare for demand variations
- **Growth Forecasting**: Project future system needs

### Real-Time Applications
- **Dynamic Pricing**: Adjust pricing based on predicted demand
- **User Notifications**: Alert users about availability
- **Route Optimization**: Suggest optimal trip timing

## Technical Details

### Data Size Considerations
- **Individual Records**: ~50M+ rides across years
- **Hourly Aggregation**: ~50K+ hours
- **Daily Aggregation**: ~2K+ days
- **Memory Usage**: ~2-4GB for recent years

### Computational Requirements
- **CPU**: Multi-core recommended for ensemble models
- **RAM**: 8GB+ for full dataset processing
- **GPU**: Optional, for deep learning acceleration
- **Storage**: ~10GB for data and models

### Scalability
- **Dask Integration**: Available for larger datasets
- **Incremental Learning**: Update models with new data
- **Distributed Computing**: Scale across multiple machines

---

*For questions or improvements, please refer to the code documentation or create an issue.* 