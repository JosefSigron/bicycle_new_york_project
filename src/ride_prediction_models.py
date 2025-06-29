import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning libraries
try:
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Dense, LSTM, GRU, Dropout, Conv1D, MaxPooling1D, Flatten
    from keras.optimizers import Adam
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Deep learning models will be skipped.")
    DEEP_LEARNING_AVAILABLE = False

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class RidePredictionModels:
    """
    Collection of machine learning models for predicting ride counts and durations
    """
    
    def __init__(self, save_plots=True, plot_dir="results/ride_prediction"):
        self.models = {}
        self.scalers = {}
        self.results = []
        self.save_plots = save_plots
        self.plot_dir = plot_dir
        
        # Create plot directory if it doesn't exist
        if self.save_plots:
            os.makedirs(self.plot_dir, exist_ok=True)
        
    def save_or_show_plot(self, filename=None):
        """
        Save plot to file or show it based on configuration
        """
        if self.save_plots and filename:
            filepath = os.path.join(self.plot_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {filepath}")
        else:
            plt.show()
        plt.close()
    
    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Evaluate model performance
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        results = {
            'model': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return results
    
    def prepare_data_for_ml(self, X, y, test_size=0.2, scale=True, time_series_split=False):
        """
        Prepare data for machine learning
        """
        # Handle missing values
        X_clean = X.fillna(X.mean())
        
        if time_series_split:
            # For time series, use chronological split
            split_idx = int(len(X_clean) * (1 - test_size))
            X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y, test_size=test_size, random_state=42
            )
        
        if scale:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            return X_train_scaled, X_test_scaled, y_train, y_test, scaler
        else:
            return X_train, X_test, y_train, y_test, None
    
    def train_traditional_models(self, X_train, X_test, y_train, y_test, target_name):
        """
        Train traditional machine learning models
        """
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0)
        }
        
        results = []
        
        for name, model in models.items():
            print(f"Training {name} for {target_name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            train_results = self.evaluate_model(y_train, y_pred_train, f"{name}_train")
            test_results = self.evaluate_model(y_test, y_pred_test, f"{name}_test")
            
            # Store results
            results.extend([train_results, test_results])
            
            # Store model
            self.models[f"{target_name}_{name}"] = model
            
            print(f"  Test RMSE: {test_results['rmse']:.2f}, R²: {test_results['r2']:.3f}")
        
        return pd.DataFrame(results)
    
    def prepare_sequences_for_lstm(self, data, target_col, sequence_length=24, features=None):
        """
        Prepare sequential data for LSTM models
        """
        if features is None:
            features = [col for col in data.columns if col != target_col and 'timestamp' not in col and 'date' not in col]
        
        # Filter features to only include numeric columns
        numeric_features = []
        for col in features:
            if col in data.columns:
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(data[col]):
                    numeric_features.append(col)
                else:
                    print(f"Skipping non-numeric column for LSTM: {col}")
        
        if len(numeric_features) == 0:
            raise ValueError("No numeric features available for LSTM")
        
        print(f"Using {len(numeric_features)} numeric features for LSTM")
        
        # Sort by time
        time_col = 'hour_timestamp' if 'hour_timestamp' in data.columns else data.columns[0]
        data_sorted = data.sort_values(time_col)
        
        X, y = [], []
        
        for i in range(sequence_length, len(data_sorted)):
            # Features for the sequence - ensure all numeric
            sequence_data = data_sorted[numeric_features].iloc[i-sequence_length:i]
            
            # Convert any remaining non-numeric data
            sequence_numeric = sequence_data.select_dtypes(include=[np.number])
            
            # Fill any NaN values
            sequence_filled = sequence_numeric.fillna(0)
            
            X.append(sequence_filled.values)
            # Target value
            y.append(data_sorted[target_col].iloc[i])
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
    
    def train_deep_learning_models(self, X_train, X_test, y_train, y_test, target_name, sequence_data=None):
        """
        Train deep learning models (if available)
        """
        if not DEEP_LEARNING_AVAILABLE:
            print("Deep learning models not available (TensorFlow not installed)")
            return pd.DataFrame()
        
        results = []
        
        # Simple feedforward neural network
        print(f"Training Neural Network for {target_name}...")
        
        model_nn = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model_nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train
        history = model_nn.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        # Predictions
        y_pred_train = model_nn.predict(X_train, verbose=0).flatten()
        y_pred_test = model_nn.predict(X_test, verbose=0).flatten()
        
        # Evaluate
        train_results = self.evaluate_model(y_train, y_pred_train, "Neural_Network_train")
        test_results = self.evaluate_model(y_test, y_pred_test, "Neural_Network_test")
        
        results.extend([train_results, test_results])
        self.models[f"{target_name}_Neural_Network"] = model_nn
        
        print(f"  Test RMSE: {test_results['rmse']:.2f}, R²: {test_results['r2']:.3f}")
        
        # LSTM model (if sequence data is provided)
        if sequence_data is not None:
            print(f"Training LSTM for {target_name}...")
            
            X_seq, y_seq = sequence_data
            
            # Split sequential data
            split_idx = int(len(X_seq) * 0.8)
            X_seq_train, X_seq_test = X_seq[:split_idx], X_seq[split_idx:]
            y_seq_train, y_seq_test = y_seq[:split_idx], y_seq[split_idx:]
            
            model_lstm = Sequential([
                LSTM(50, return_sequences=True, input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            
            # Train
            history_lstm = model_lstm.fit(
                X_seq_train, y_seq_train,
                epochs=30,
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
            
            # Predictions
            y_pred_lstm_train = model_lstm.predict(X_seq_train, verbose=0).flatten()
            y_pred_lstm_test = model_lstm.predict(X_seq_test, verbose=0).flatten()
            
            # Evaluate
            train_results_lstm = self.evaluate_model(y_seq_train, y_pred_lstm_train, "LSTM_train")
            test_results_lstm = self.evaluate_model(y_seq_test, y_pred_lstm_test, "LSTM_test")
            
            results.extend([train_results_lstm, test_results_lstm])
            self.models[f"{target_name}_LSTM"] = model_lstm
            
            print(f"  LSTM Test RMSE: {test_results_lstm['rmse']:.2f}, R²: {test_results_lstm['r2']:.3f}")
        
        return pd.DataFrame(results)
    
    def plot_feature_importance(self, model, feature_names, model_name, top_n=20):
        """
        Plot and save feature importance for tree-based models
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title(f'Top {top_n} Feature Importance - {model_name}')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            filename = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
            self.save_or_show_plot(filename)
            
            return importance_df
        else:
            print(f"Feature importance not available for {model_name}")
            return None
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name, target_name):
        """
        Plot predictions vs actual values
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(f'{model_name} - {target_name}: Predictions vs Actual')
        
        # Add R² score to plot
        r2 = r2_score(y_true, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        filename = f"predictions_vs_actual_{model_name.replace(' ', '_').lower()}_{target_name}.png"
        self.save_or_show_plot(filename)
    
    def plot_time_series_comparison(self, data_clean, y_true, y_pred, model_name, target_name, data_type='hourly'):
        """
        Plot actual vs predicted values over time
        """
        plt.figure(figsize=(15, 8))
        
        # Prepare time index
        if data_type == 'hourly':
            time_col = 'hour_timestamp'
            title_freq = 'Hourly'
        else:
            time_col = 'date'
            title_freq = 'Daily'
        
        # Get time index for test data (assuming test data is the last portion)
        n_test = len(y_true)
        time_index = data_clean[time_col].iloc[-n_test:].reset_index(drop=True)
        
        # Plot actual vs predicted
        plt.plot(time_index, y_true, label='Actual', color='blue', alpha=0.7, linewidth=2)
        plt.plot(time_index, y_pred, label='Predicted', color='red', alpha=0.7, linewidth=2)
        
        plt.xlabel('Time')
        plt.ylabel(target_name.replace('_', ' ').title())
        plt.title(f'{title_freq} {target_name.replace("_", " ").title()} - Actual vs Predicted ({model_name})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Add performance metrics as text
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        metrics_text = f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
        plt.text(0.02, 0.98, metrics_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        filename = f"time_series_comparison_{model_name.replace(' ', '_').lower()}_{target_name}_{data_type}.png"
        self.save_or_show_plot(filename)

    def plot_residuals_over_time(self, data_clean, y_true, y_pred, model_name, target_name, data_type='hourly'):
        """
        Plot residuals (prediction errors) over time
        """
        plt.figure(figsize=(15, 6))
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Prepare time index
        if data_type == 'hourly':
            time_col = 'hour_timestamp'
            title_freq = 'Hourly'
        else:
            time_col = 'date'
            title_freq = 'Daily'
        
        # Get time index for test data
        n_test = len(y_true)
        time_index = data_clean[time_col].iloc[-n_test:].reset_index(drop=True)
        
        # Plot residuals
        plt.scatter(time_index, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        
        plt.xlabel('Time')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.title(f'{title_freq} {target_name.replace("_", " ").title()} - Residuals Over Time ({model_name})')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Add statistics
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        stats_text = f'Mean Residual = {mean_residual:.2f}\nStd Residual = {std_residual:.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=10)
        
        plt.tight_layout()
        
        filename = f"residuals_over_time_{model_name.replace(' ', '_').lower()}_{target_name}_{data_type}.png"
        self.save_or_show_plot(filename)

    def create_comprehensive_prediction_plots(self, data_clean, y_true, y_pred, model_name, target_name, data_type='hourly'):
        """
        Create a comprehensive set of plots for prediction analysis
        """
        print(f"Creating comprehensive plots for {model_name} - {target_name}")
        
        # 1. Predictions vs Actual scatter plot
        self.plot_predictions_vs_actual(y_true, y_pred, model_name, target_name)
        
        # 2. Time series comparison
        self.plot_time_series_comparison(data_clean, y_true, y_pred, model_name, target_name, data_type)
        
        # 3. Residuals over time
        self.plot_residuals_over_time(data_clean, y_true, y_pred, model_name, target_name, data_type)

    def plot_model_comparison(self, results_df, target_name, data_type='hourly'):
        """
        Plot comparison of all models' performance
        """
        # Filter test results only
        test_results = results_df[results_df['model'].str.contains('_test')].copy()
        test_results['model_clean'] = test_results['model'].str.replace('_test', '')
        test_results = test_results.sort_values('rmse')
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. RMSE comparison
        sns.barplot(data=test_results, x='rmse', y='model_clean', ax=ax1, palette='viridis')
        ax1.set_title('RMSE Comparison (Lower is Better)')
        ax1.set_xlabel('RMSE')
        ax1.set_ylabel('Model')
        
        # 2. R² comparison
        sns.barplot(data=test_results, x='r2', y='model_clean', ax=ax2, palette='plasma')
        ax2.set_title('R² Comparison (Higher is Better)')
        ax2.set_xlabel('R²')
        ax2.set_ylabel('Model')
        
        # 3. MAE comparison
        sns.barplot(data=test_results, x='mae', y='model_clean', ax=ax3, palette='cividis')
        ax3.set_title('MAE Comparison (Lower is Better)')
        ax3.set_xlabel('MAE')
        ax3.set_ylabel('Model')
        
        # 4. Performance scatter plot (RMSE vs R²)
        ax4.scatter(test_results['rmse'], test_results['r2'], s=100, alpha=0.7, c=range(len(test_results)), cmap='rainbow')
        for i, row in test_results.iterrows():
            ax4.annotate(row['model_clean'], (row['rmse'], row['r2']), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, alpha=0.8)
        ax4.set_xlabel('RMSE')
        ax4.set_ylabel('R²')
        ax4.set_title('RMSE vs R² (Bottom Right is Best)')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Model Performance Comparison - {target_name.replace("_", " ").title()} ({data_type.title()})', 
                    fontsize=16, y=0.98)
        plt.tight_layout()
        
        filename = f"model_comparison_{target_name}_{data_type}.png"
        self.save_or_show_plot(filename)
        
        return test_results

    def time_series_cross_validation(self, X, y, model, n_splits=5):
        """
        Perform time series cross validation
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        rmse_scores = np.sqrt(-scores)
        
        return {
            'mean_rmse': rmse_scores.mean(),
            'std_rmse': rmse_scores.std(),
            'scores': rmse_scores
        }
    
    def create_ensemble_model(self, models, X_test, y_test, weights=None):
        """
        Create ensemble predictions from multiple models
        """
        predictions = []
        
        for model_name, model in models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X_test)
                if len(pred.shape) > 1:
                    pred = pred.flatten()
                predictions.append(pred)
        
        if not predictions:
            print("No valid models for ensemble")
            return None
        
        predictions = np.array(predictions)
        
        if weights is None:
            # Equal weights
            ensemble_pred = np.mean(predictions, axis=0)
        else:
            # Weighted average
            ensemble_pred = np.average(predictions, axis=0, weights=weights)
        
        # Evaluate ensemble
        ensemble_results = self.evaluate_model(y_test, ensemble_pred, "Ensemble")
        
        return ensemble_pred, ensemble_results


def main_prediction_pipeline(preprocessor, target='ride_count', data_type='hourly'):
    """
    Main pipeline for running all prediction models
    """
    print(f"=== PREDICTION PIPELINE FOR {target.upper()} ===")
    
    # Get data
    X, y, data_clean = preprocessor.get_feature_target_split(data_type=data_type, target=target)
    
    print(f"Data shape: {X.shape}")
    print(f"Target range: {y.min():.2f} to {y.max():.2f}")
    
    # Initialize model trainer with plot saving enabled
    model_trainer = RidePredictionModels(save_plots=True, plot_dir="results/ride_prediction")
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler = model_trainer.prepare_data_for_ml(
        X, y, test_size=0.2, scale=True, time_series_split=True
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train traditional models
    print("\n--- TRADITIONAL MACHINE LEARNING MODELS ---")
    traditional_results = model_trainer.train_traditional_models(
        X_train, X_test, y_train, y_test, target
    )
    
    # Train deep learning models
    print("\n--- DEEP LEARNING MODELS ---")
    
    # Prepare sequence data for LSTM
    sequence_data = None
    if data_type == 'hourly':
        try:
            X_seq, y_seq = model_trainer.prepare_sequences_for_lstm(
                data_clean, target, sequence_length=24
            )
            sequence_data = (X_seq, y_seq)
            print(f"Sequential data prepared: {X_seq.shape}")
        except Exception as e:
            print(f"Could not prepare sequential data: {e}")
    
    deep_results = model_trainer.train_deep_learning_models(
        X_train, X_test, y_train, y_test, target, sequence_data
    )
    
    # Combine results
    all_results = pd.concat([traditional_results, deep_results], ignore_index=True)
    
    # Display results
    print("\n--- MODEL COMPARISON ---")
    test_results = all_results[all_results['model'].str.contains('_test')]
    test_results = test_results.sort_values('rmse')
    print(test_results[['model', 'rmse', 'mae', 'r2']].to_string(index=False))
    
    # Create model comparison plot
    print("\n--- CREATING MODEL COMPARISON PLOT ---")
    model_trainer.plot_model_comparison(all_results, target, data_type)
    
    # Plot feature importance for best tree model
    best_tree_models = ['Random Forest', 'Gradient Boosting']
    for model_name in best_tree_models:
        model_key = f"{target}_{model_name}"
        if model_key in model_trainer.models:
            print(f"\n--- FEATURE IMPORTANCE: {model_name} ---")
            importance_df = model_trainer.plot_feature_importance(
                model_trainer.models[model_key], 
                X.columns, 
                model_name
            )
    
    # Plot predictions for best model
    best_model_name = test_results.iloc[0]['model'].replace('_test', '')
    best_model_key = f"{target}_{best_model_name}"
    
    if best_model_key in model_trainer.models:
        print(f"\n--- CREATING COMPREHENSIVE PLOTS: {best_model_name} ---")
        best_model = model_trainer.models[best_model_key]
        
        if hasattr(best_model, 'predict'):
            y_pred_best = best_model.predict(X_test)
            if len(y_pred_best.shape) > 1:
                y_pred_best = y_pred_best.flatten()
            
            # Create comprehensive prediction plots (includes time series comparison)
            model_trainer.create_comprehensive_prediction_plots(
                data_clean, y_test, y_pred_best, best_model_name, target, data_type
            )
    
    return model_trainer, all_results


if __name__ == "__main__":
    # This would be run with the preprocessed data
    from ride_prediction_preprocessing import RideDataPreprocessor
    
    # Load preprocessed data
    try:
        hourly_data = pd.read_parquet("data/processed/hourly_aggregated.parquet")
        daily_data = pd.read_parquet("data/processed/daily_aggregated.parquet")
        
        print("Using existing processed data...")
        
        # Create mock preprocessor with data
        preprocessor = RideDataPreprocessor()
        preprocessor.hourly_aggregated = hourly_data
        preprocessor.daily_aggregated = daily_data
        
    except FileNotFoundError:
        print("Processed data not found. Please run preprocessing first.")
        print("Example: python src/ride_prediction_preprocessing.py")
        exit(1)
    
    # Run prediction for ride count (hourly)
    print("=== HOURLY RIDE COUNT PREDICTION ===")
    ride_count_trainer, ride_count_results = main_prediction_pipeline(
        preprocessor, target='ride_count', data_type='hourly'
    )
    
    # Run prediction for average duration (hourly)
    print("\n\n=== HOURLY AVERAGE DURATION PREDICTION ===")
    duration_trainer, duration_results = main_prediction_pipeline(
        preprocessor, target='avg_duration', data_type='hourly'
    )
    
    # Run prediction for daily ride count
    print("\n\n=== DAILY RIDE COUNT PREDICTION ===")
    daily_count_trainer, daily_count_results = main_prediction_pipeline(
        preprocessor, target='daily_ride_count', data_type='daily'
    )
    
    print("\nPrediction pipeline complete!") 