"""
Unified Models Module for AUDUSD Prediction

This module contains all the model classes for predicting AUDUSD rates:
- BaseModel: Abstract base class defining the common interface
- ClassicalModel: Linear regression models (Linear, Ridge, Lasso)
- ArimaModel: Time series ARIMA models with walk-forward validation
- CNNModel: Deep learning CNN models for multi-horizon forecasting
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Imports for ClassicalModel - using try/except for optional dependencies
try:
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. ClassicalModel will not work.")

# Imports for ArimaModel - using try/except for optional dependencies
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not available. ArimaModel will not work.")

# Imports for CNNModel - using try/except for optional dependencies
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import backend, optimizers
    from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv1D, LSTM
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. CNNModel will not work.")


class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    
    This class defines the common interface that all model implementations
    must follow, ensuring consistency across different model types.
    """
    
    def __init__(self, data, target_col='audusd'):
        """
        Initialize the base model.
        
        Parameters:
        - data: DataFrame with time series data
        - target_col: Name of the target column to predict
        """
        self.data = data.copy()
        self.target_col = target_col
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, **kwargs):
        """Fit the model to the data. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def predict(self, **kwargs):
        """Make predictions. Must be implemented by subclasses."""
        pass
    
    def get_rmse(self):
        """Get RMSE metric. Can be overridden by subclasses."""
        if hasattr(self, 'rmse_value') and self.rmse_value is not None:
            return self.rmse_value
        else:
            print("RMSE not available. Train and evaluate the model first.")
            return None
    
    def get_mae(self):
        """Get MAE metric. Can be overridden by subclasses."""
        if hasattr(self, 'mae_value') and self.mae_value is not None:
            return self.mae_value
        else:
            print("MAE not available. Train and evaluate the model first.")
            return None


class ArimaModel(BaseModel):
    """
    ARIMA Model for time series forecasting with walk-forward validation.
    
    Features:
    - Automatic stationarity testing
    - Walk-forward validation for robust evaluation
    - Confidence interval forecasting
    - Support for exogenous variables
    """
    
    def __init__(self, data, order=(1, 1, 1), target_col='audusd'):
        """
        Initialize ARIMA model.
        
        Parameters:
        - data: DataFrame with time series data
        - order: ARIMA order (p, d, q)
        - target_col: Name of the target column to predict
        """
        super().__init__(data, target_col)
        self.order = order
        self.predictions = None
        self.test_data = None
        self.rmse_value = None
        
    def fit(self, order=None):
        """
        Fit the ARIMA model.
        
        Parameters:
        - order: ARIMA order (p,d,q). If None, uses self.order
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ArimaModel")
            
        if order is None:
            order = self.order
            
        self.model = ARIMA(self.data[self.target_col], order=order)
        self.model = self.model.fit()
        self.is_fitted = True

    def predict(self, steps=5):
        """
        Generate forecasts.
        
        Parameters:
        - steps: Number of steps to forecast
        
        Returns:
        - Forecasted values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.forecast(steps=steps)

    def summary(self):
        """Get model summary if fitted."""
        return self.model.summary() if self.model else None

    def walk_forward_validation(self, test_size=50, order=None, exog_vars=None, verbose=True):
        """
        Perform walk-forward validation for ARIMA model
        
        Parameters:
        - test_size: Number of observations to use for testing
        - order: ARIMA order (p,d,q). If None, uses self.order
        - exog_vars: List of exogenous variable names to include
        - verbose: Whether to print predictions vs actual values
        
        Returns:
        - predictions: List of forecasted values
        - test: Test data series
        - rmse: Root Mean Square Error
        """
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ArimaModel")
            
        start_time = time.time()
        
        if order is None:
            order = self.order
            
        # Prepare data
        X = self.data[self.target_col].copy()
        
        # Handle exogenous variables if provided
        exog_data = None
        if exog_vars:
            exog_data = self.data[exog_vars].copy()
            exog_data = exog_data.loc[X.index]
            
        # Split data
        size = len(X) - test_size
        train, test = X[:size], X[size:]
        
        if exog_data is not None:
            exog_train, exog_test = exog_data[:size], exog_data[size:]
        
        # Initialize
        history = [x for x in train]
        exog_history = exog_train.values.tolist() if exog_data is not None else None
        predictions = []
        
        print(f"Starting walk-forward validation with {len(test)} test points...")
        
        # Walk-forward validation
        for t in range(len(test)):
            try:
                # Fit model
                if exog_data is not None:
                    model = ARIMA(history, order=order, exog=exog_history, trend='n')
                else:
                    model = ARIMA(history, order=order, trend='n')
                    
                model_fit = model.fit()
                
                # Make prediction
                if exog_data is not None:
                    exog_forecast = exog_test.iloc[t:t+1].values
                    output = model_fit.forecast(exog=exog_forecast)
                else:
                    output = model_fit.forecast()
                    
                yhat = output[0] if hasattr(output, '__len__') else output
                predictions.append(yhat)
                
                # Update history
                obs = test.iloc[t]
                history.append(obs)
                
                if exog_data is not None:
                    exog_history.append(exog_test.iloc[t].values.tolist())
                
                if verbose:
                    print(f'Step {t+1}: predicted={yhat:.6f}, expected={obs:.6f}')
                    
            except Exception as e:
                print(f"Error at step {t+1}: {e}")
                # Use last prediction or naive forecast
                if predictions:
                    predictions.append(predictions[-1])
                else:
                    predictions.append(history[-1])
        
        # Store results
        self.predictions = predictions
        self.test_data = test
        self.rmse_value = np.sqrt(mean_squared_error(test, predictions))
        
        elapsed_time = time.time() - start_time
        print(f"Walk-forward validation completed in {elapsed_time:.2f} seconds")
        print(f'Test RMSE: {self.rmse_value:.6f}')
        
        return predictions, test, self.rmse_value

    def plot_forecast_results(self, figsize=(20, 5)):
        """
        Plot the actual vs predicted values from walk-forward validation
        """
        if self.predictions is None or self.test_data is None:
            print("No forecast results available. Run walk_forward_validation first.")
            return
            
        plt.figure(figsize=figsize)
        plt.plot(self.test_data.index, self.test_data, color='blue', label='Actual')
        plt.plot(self.test_data.index, self.predictions, color='red', label='Predictions')
        plt.legend()
        plt.title(f'ARIMA Walk-Forward Validation Results (RMSE: {self.rmse_value:.6f})')
        plt.xlabel('Date')
        plt.ylabel(f'{self.target_col.upper()}')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def forecast_with_confidence(self, forecast_horizon=10, history_window=30, figsize=(10, 6)):
        """
        Generate forecasts with confidence intervals and plot them along with recent history
        
        Parameters:
        - forecast_horizon: Number of periods to forecast into the future
        - history_window: Number of recent periods to show in the plot
        - figsize: Figure size for the plot
        
        Returns:
        - mean_forecast: Forecasted values
        - conf_int: Confidence intervals
        - history_data: Recent historical data used in the plot
        """
        if self.model is None:
            print("Model not fitted. Please call fit() first.")
            return None, None, None
            
        # Generate forecast
        forecast_obj = self.model.get_forecast(steps=forecast_horizon)
        mean_forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int()
        
        # Get recent history for plotting using original dataframe index
        history_data = self.data[self.target_col].iloc[-history_window:]
        history_index = history_data.index
        
        # Create future datetime index for forecasts
        # Get the frequency of the original data
        last_date = self.data.index[-1]
        if len(self.data.index) > 1:
            # Try to infer the frequency from the data
            time_diff = self.data.index[-1] - self.data.index[-2]
            forecast_dates = [last_date + (i + 1) * time_diff for i in range(forecast_horizon)]
        else:
            # Fallback: assume daily frequency
            import pandas as pd
            forecast_dates = pd.date_range(start=last_date, periods=forecast_horizon + 1, freq='D')[1:]
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Plot last N days of history
        plt.plot(history_index, history_data, label=f'Last {history_window} Days History', color='blue', linewidth=2)
        
        # Plot forecast mean
        plt.plot(forecast_dates, mean_forecast, label='Forecast', color='orange', linewidth=2)
        
        # Plot confidence intervals
        plt.fill_between(forecast_dates,
                         conf_int.iloc[:, 0],  # lower bound
                         conf_int.iloc[:, 1],  # upper bound
                         color='orange', alpha=0.3, label='95% Confidence Interval')
        
        plt.legend()
        plt.title(f'ARIMA Forecast with 95% Confidence Interval (Last {history_window} Days + {forecast_horizon} Day Forecast)')
        plt.xlabel('Date')
        plt.ylabel(f'{self.target_col.upper()}')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Print forecast summary with dates
        print(f"Forecast Summary for next {forecast_horizon} periods:")
        print("-" * 70)
        for i, (date, mean_val, lower, upper) in enumerate(zip(forecast_dates, mean_forecast, conf_int.iloc[:, 0], conf_int.iloc[:, 1])):
            print(f"{date.strftime('%Y-%m-%d')}: {mean_val:.6f} (95% CI: {lower:.6f} - {upper:.6f})")
        
        return mean_forecast, conf_int, history_data


class ClassicalModel(BaseModel):
    """
    Classical machine learning models (Linear Regression, Ridge, Lasso)
    """
    def __init__(self, data, target_col='audusd', model_type='linear', test_size=0.2, 
                 random_state=42, scale_features=True):
        super().__init__(data, target_col)
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.scale_features = scale_features
        self.scaler = None
        self.rmse_value = None
        
    def _initialize_model(self, **kwargs):
        """Initialize the model based on type"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ClassicalModel")
            
        if self.model_type == 'linear':
            self.model = LinearRegression(**kwargs)
        elif self.model_type == 'ridge':
            alpha = kwargs.get('alpha', 1.0)
            self.model = Ridge(alpha=alpha, **{k: v for k, v in kwargs.items() if k != 'alpha'})
        elif self.model_type == 'lasso':
            alpha = kwargs.get('alpha', 1.0)
            self.model = Lasso(alpha=alpha, **{k: v for k, v in kwargs.items() if k != 'alpha'})
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def fit(self, **model_params):
        """Fit the model with proper preprocessing"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ClassicalModel")
            
        # Initialize model with parameters
        self._initialize_model(**model_params)
        
        # Prepare data
        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]
        
        # Handle missing values
        initial_shape = X.shape[0]
        X = X.dropna()
        y = y[X.index]
        
        if X.shape[0] < initial_shape:
            print(f"Dropped {initial_shape - X.shape[0]} rows due to missing values")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            shuffle=False  # Important for time series data
        )
        
        # Scale features if requested
        if self.scale_features:
            self.scaler = StandardScaler()
            X_train_processed = self.scaler.fit_transform(self.X_train)
            X_test_processed = self.scaler.transform(self.X_test)
        else:
            X_train_processed = self.X_train.values
            X_test_processed = self.X_test.values
        
        # Fit model
        self.model.fit(X_train_processed, self.y_train)
        self.is_fitted = True
        
        # Store processed data
        self.X_train_processed = X_train_processed
        self.X_test_processed = X_test_processed
        
        print(f"Model fitted successfully!")
        print(f"Training samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")
        
        return self
    
    def predict(self, X=None):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if X is None:
            # Predict on test set
            return self.model.predict(self.X_test_processed)
        else:
            # Predict on new data
            if self.scale_features and self.scaler is not None:
                X_processed = self.scaler.transform(X)
            else:
                X_processed = X
            return self.model.predict(X_processed)
    
    def evaluate(self):
        """Evaluate model performance"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ClassicalModel")
            
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train_processed)
        y_test_pred = self.model.predict(self.X_test_processed)
        
        # Metrics
        metrics = {
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred)
        }
        
        # Store RMSE
        self.rmse_value = metrics['test_rmse']
        
        # Print results
        print(f"Training RMSE: {metrics['train_rmse']:.4f}")
        print(f"Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"Training MAE: {metrics['train_mae']:.4f}")
        print(f"Test MAE: {metrics['test_mae']:.4f}")
        
        # Check for overfitting
        r2_diff = metrics['train_r2'] - metrics['test_r2']
        if r2_diff > 0.1:
            print(f"⚠️ Potential overfitting detected (R² difference: {r2_diff:.4f})")
        
        return metrics
    
    def cross_validate(self, cv=5):
        """Perform cross-validation"""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for ClassicalModel")
            
        if not self.is_fitted:
            raise ValueError("Model must be fitted before cross-validation")
        
        X = self.data.drop(columns=[self.target_col]).dropna()
        y = self.data[self.target_col][X.index]
        
        if self.scale_features:
            X = StandardScaler().fit_transform(X)
        
        # Create fresh model for CV
        if self.model_type == 'linear':
            cv_model = LinearRegression()
        elif self.model_type == 'ridge':
            cv_model = Ridge(alpha=self.model.alpha)
        elif self.model_type == 'lasso':
            cv_model = Lasso(alpha=self.model.alpha)
        
        cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring='r2')
        
        print(f"\n {cv}-Fold Cross-Validation Results:")
        print(f"Mean R²: {cv_scores.mean():.4f} (±{cv_scores.std() * 2:.4f})")
        print(f"Individual scores: {cv_scores}")
        
        return cv_scores
    
    def plot_results(self):
        """Plot predictions vs actual values"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        y_train_pred = self.model.predict(self.X_train_processed)
        y_test_pred = self.model.predict(self.X_test_processed)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training set
        ax1.scatter(self.y_train, y_train_pred, alpha=0.6, label='Training')
        ax1.plot([self.y_train.min(), self.y_train.max()], 
                [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Training Set: Predicted vs Actual')
        ax1.legend()
        
        # Test set
        ax2.scatter(self.y_test, y_test_pred, alpha=0.6, color='orange', label='Test')
        ax2.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Predicted Values')
        ax2.set_title('Test Set: Predicted vs Actual')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def get_feature_importance(self):
        """Get feature coefficients/importance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        feature_names = self.X_train.columns
        coefficients = self.model.coef_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        print("\n📈 Feature Importance (Top 10):")
        print(importance_df.head(10))
        
        return importance_df


class NeuralNetworkBaseModel(BaseModel):
    """
    Base class for neural network models (CNN, LSTM) with shared logic.
    """
    def __init__(self, data, target_col='audusd', train_ratio=0.80):
        super().__init__(data, target_col)
        self.train_ratio = train_ratio
        self.history = None
        self.train = None
        self.test = None
        self.index_train = None
        self.index_test = None
        self.train_x = None
        self.train_y = None
        self.y0_train = None
        self.test_x = None
        self.test_y = None
        self.y0_test = None
        self.train_pred = None
        self.train_true = None
        self.test_pred = None
        self.test_true = None
        self.mae_test = None
        self.rmse_test = None
        self.seq_len = 20
        self.forecast_horizon = 30

    def split_data(self):
        train_size = int(len(self.data) * self.train_ratio)
        self.train = self.data.iloc[:train_size, :]
        self.test = self.data.iloc[train_size:, :]
        self.index_train = self.train.index
        self.index_test = self.test.index
        print(f"Train size: {len(self.train)}, Test size: {len(self.test)}")

    def seq_and_norm(self, dataframe, seq_len, forecast_horizon):
        eps = 1e-8
        sequence_x, target, initial_y = [], [], []
        n = len(dataframe)
        for start in range(0, n - seq_len - forecast_horizon + 1):
            x = dataframe.iloc[start:start + seq_len, :].copy()
            target_col_idx = dataframe.columns.get_loc(self.target_col)
            y = dataframe.iloc[start + 1:start + 1 + forecast_horizon, target_col_idx].copy()
            x = x.ffill().bfill()
            y = y.ffill().bfill()
            x0 = x.iloc[0, :].replace([0, np.inf, -np.inf], eps)
            y0 = y.iloc[0]
            if (y0 == 0) or np.isinf(y0) or np.isnan(y0):
                y0 = eps
            x_norm = x.divide(x0, axis='columns') - 1.0
            y_norm = (y / y0) - 1.0
            if not np.isfinite(x_norm.values).all() or not np.isfinite(y_norm.values).all():
                continue
            sequence_x.append(x_norm.values)
            target.append(y_norm.values)
            initial_y.append(np.full(shape=(forecast_horizon,), fill_value=y0))
        return np.array(sequence_x), np.array(target), np.array(initial_y)

    def prepare_data(self, seq_len=20, forecast_horizon=30):
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        print(f"seq_len={seq_len}, forecast_horizon={forecast_horizon}")
        if self.train is None or self.test is None:
            self.split_data()
        self.train_x, self.train_y, self.y0_train = self.seq_and_norm(self.train, seq_len, forecast_horizon)
        self.test_x, self.test_y, self.y0_test = self.seq_and_norm(self.test, seq_len, forecast_horizon)
        print("Shapes ->", "train_x:", self.train_x.shape, "train_y:", self.train_y.shape, "y0_train:", self.y0_train.shape)
        print("Shapes ->", "test_x:", self.test_x.shape, "test_y:", self.test_y.shape, "y0_test:", self.y0_test.shape)

    def inverse_transform(self, y_norm, y0):
        return y_norm * y0 + y0

    def per_horizon_metrics(self, y_true, y_pred):
        mae_list, rmse_list = [], []
        H = y_true.shape[1]
        for h in range(H):
            mae_list.append(mean_absolute_error(y_true[:, h], y_pred[:, h]))
            rmse_list.append(np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h])))
        return np.array(mae_list), np.array(rmse_list)

    def rmspe(self, y_true, y_pred, eps=1e-8):
        return np.sqrt(np.mean(np.square((y_true - y_pred) / (np.abs(y_true) + eps)))) * 100.0

    def predict_and_evaluate(self):
        if not self.is_fitted:
            print("Model not trained. Call train_model() first.")
            return
        train_pred_norm = self.model.predict(self.train_x, verbose=0)
        test_pred_norm = self.model.predict(self.test_x, verbose=0)
        self.train_pred = self.inverse_transform(train_pred_norm, self.y0_train)
        self.train_true = self.inverse_transform(self.train_y, self.y0_train)
        self.test_pred = self.inverse_transform(test_pred_norm, self.y0_test)
        self.test_true = self.inverse_transform(self.test_y, self.y0_test)
        self.mae_test, self.rmse_test = self.per_horizon_metrics(self.test_true, self.test_pred)
        print("Test MAE per horizon:", self.mae_test)
        print("Test RMSE per horizon:", self.rmse_test)
        self.rmse_value = self.rmse_test[0] if len(self.rmse_test) > 0 else None
        rmspe_h1 = self.rmspe(self.test_true[:, 0], self.test_pred[:, 0])
        print(f"RMSPE (H+1): {rmspe_h1:.4f}%")

    def plot_results(self, N=100, horizon=10):
        if self.test_pred is None:
            print("No predictions available. Call predict_and_evaluate() first.")
            return
        plt.figure(figsize=(12, 5))
        plt.plot(self.test_true[-N:, horizon], label=f'Actual H+{horizon+1}')
        plt.plot(self.test_pred[-N:, horizon], label=f'Predicted H+{horizon+1}')
        plt.title(f'H+{horizon+1} Forecast (last {N} samples) | MAE={self.mae_test[horizon]:.6f}, RMSE={self.rmse_test[horizon]:.6f}')
        plt.xlabel('Sample')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_metrics_by_horizon(self):
        if self.mae_test is None or self.rmse_test is None:
            print("No metrics available. Call predict_and_evaluate() first.")
            return
        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(1, self.forecast_horizon + 1), self.mae_test, marker='o', label='MAE')
        plt.plot(np.arange(1, self.forecast_horizon + 1), self.rmse_test, marker='x', label='RMSE')
        plt.title('Test Metrics by Horizon')
        plt.xlabel('Horizon (steps ahead)')
        plt.ylabel('Error (original scale)')
        plt.tight_layout()
        plt.show()

    def get_rmse(self, horizon=0):
        if self.rmse_test is None:
            print("No metrics available. Call predict_and_evaluate() first.")
            return None
        return self.rmse_test[horizon]

    def get_mae(self, horizon=0):
        if self.mae_test is None:
            print("No metrics available. Call predict_and_evaluate() first.")
            return None
        return self.mae_test[horizon]


class CNNModel(NeuralNetworkBaseModel):
    """
    Convolutional Neural Network model for multi-horizon time series forecasting
    """
    def build_model(self):
        """Build the CNN model architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNNModel")
            
        backend.clear_session()
        lag = self.train_x.shape[1]          # seq_len
        n_features = self.train_x.shape[2]   # number of columns in data

        # Optional: swish activation
        @tf.keras.utils.register_keras_serializable()
        def swish(x):
            return tf.nn.swish(x)

        self.model = Sequential([
            Conv1D(filters=32, kernel_size=3, padding='same', activation=swish, input_shape=(lag, n_features)),
            Conv1D(filters=32, kernel_size=3, padding='same', activation=swish),
            Flatten(),
            Dense(32, activation=lambda x: tf.sin(x)),
            Dropout(0.4),
            Dense(self.forecast_horizon)  # multi-horizon output
        ])

        self.model.compile(optimizer=optimizers.Adam(learning_rate=5e-4), loss='mse', metrics=['mae'])
        self.model.summary()

    def train_model(self, epochs=2048, batch_size=256, validation_split=0.2, patience=50, verbose=1):
        """Train the CNN model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for CNNModel")
            
        if self.model is None:
            print("Model not built. Call build_model() first.")
            return
        
        start_time = time.time()
        self.history = self.model.fit(
            self.train_x, self.train_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            callbacks=[EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, mode='min')]
        )

        # Evaluate on normalized scale (loss/mae are in normalized space)
        val = self.model.evaluate(self.test_x, self.test_y, verbose=0)
        print({"test_loss": val[0], "test_mae": val[1]})
        print("Training time (s):", time.time() - start_time)
        self.is_fitted = True


    def fit(self, seq_len=20, forecast_horizon=30, epochs=2048, batch_size=256, validation_split=0.2, patience=50, verbose=1):
        print("=== Starting CNN Model Training Pipeline ===")
        print("1. Preparing data...")
        self.prepare_data(seq_len, forecast_horizon)
        print("2. Building model...")
        self.build_model()
        print("3. Training model...")
        self.train_model(epochs, batch_size, validation_split, patience, verbose)
        print("4. Generating predictions and evaluating...")
        self.predict_and_evaluate()
        print("=== Training Pipeline Complete ===")

    def predict(self, steps=None, horizon=0):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if self.test_pred is None:
            print("No predictions available. Call predict_and_evaluate() first.")
            return None
        return self.test_pred[:, horizon]


class LSTMModel(NeuralNetworkBaseModel):
    """
    Long Short-Term Memory (LSTM) model for multi-horizon time series forecasting
    """
    def build_model(self):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTMModel")
        backend.clear_session()
        lag = self.train_x.shape[1]
        n_features = self.train_x.shape[2]
        self.model = Sequential([
            LSTM(128, activation='tanh', input_shape=(lag, n_features), return_sequences=True, dropout=0.5, recurrent_dropout=0.0),
            LSTM(128, activation='tanh', return_sequences=True, dropout=0.8),
            LSTM(128, activation='tanh', return_sequences=True, dropout=0.8),
            LSTM(128, activation='tanh', return_sequences=True, dropout=0.8),
            LSTM(128, activation='tanh', return_sequences=True, dropout=0.8),
            LSTM(128, activation='tanh', return_sequences=True, dropout=0.8),
            LSTM(128, activation='tanh', return_sequences=True, dropout=0.8),
            Flatten(),
            Dense(100, activation='tanh'),
            Dropout(0.8),
            Dense(32, activation='tanh'),
            Dropout(0.8),
            Dense(1)
        ])
        self.model.compile(optimizer=optimizers.Adam(learning_rate=5e-4), loss='mse', metrics=['mae'])
        self.model.summary()

    def train_model(self, epochs=2048, batch_size=256, validation_split=0.2, patience=50, verbose=1):
        if self.model is None:
            raise ValueError("Model is not built yet")
        callbacks_list = [
            ModelCheckpoint(
                filepath="LSTM-weights-best.h5",
                monitor='val_loss',
                save_best_only=True,
                mode='auto',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1
            )
        ]
        self.history = self.model.fit(self.train_x, self.train_y, validation_data=(self.test_x, self.test_y),
                                      epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)

    def fit(self, seq_len=20, forecast_horizon=30, epochs=2048, batch_size=256, validation_split=0.2, patience=50, verbose=1):
        print("=== Starting LSTM Model Training Pipeline ===")
        print("1. Preparing data...")
        self.prepare_data(seq_len, forecast_horizon)
        print("2. Building model...")
        self.build_model()
        print("3. Training model...")
        self.train_model(epochs, batch_size, validation_split, patience, verbose)
        print("4. Generating predictions and evaluating...")
        self.predict_and_evaluate()
        print("=== Training Pipeline Complete ===")

    def predict(self, steps=None, horizon=0):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        if self.test_pred is None:
            print("No predictions available. Call predict_and_evaluate() first.")
            return None
        return self.test_pred[:, horizon]

if __name__ == "__main__":
    print("Models module loaded successfully!")
    print("Available models:")
    print("- ArimaModel: Time series ARIMA models")
    print("- ClassicalModel: Linear regression models (Linear, Ridge, Lasso)")
    print("- CNNModel: Deep learning CNN models for multi-horizon forecasting")