#!/usr/bin/env python3
"""
Generated script to create Tufte-style visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Set random seeds
np.random.seed(42)
try:
    import tensorflow as tf
    tf.random.set_seed(42)
except ImportError:
    tf = None
except:
    pass

# Tufte-style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Times'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

images_dir = Path("images")
images_dir.mkdir(exist_ok=True)

# Update all savefig calls to use images_dir
import matplotlib.pyplot as plt
original_savefig = plt.savefig

def savefig_tufte(filename, **kwargs):
    """Wrapper to save figures in images directory with Tufte style"""
    if not str(filename).startswith('/') and not str(filename).startswith('images/'):
        filename = images_dir / filename
    original_savefig(filename, **kwargs)
    logger.info(f"Saved: {filename}")

plt.savefig = savefig_tufte

# Code blocks from article
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Palatino', 'Times New Roman', 'Times'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'normal',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.5,
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'text.color': '#333333',
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Load production data
data_path = Path("../../geospatial/datasets/pr_OK.csv")
df = pd.read_csv(data_path)

pr_cols = [col for col in df.columns if col.isdigit()]

# Melt to long format
df_long = df.melt(
    id_vars=['State', 'MSN'],
    value_vars=pr_cols,
    var_name='Year',
    value_name='Value'
)

df_long['Year'] = pd.to_datetime(df_long['Year'], format='%Y')
df_long = df_long.sort_values('Year')

# Get total production
total_prod = df_long[df_long['MSN'].str.contains('TOT', na=False)].copy()
total_prod = total_prod.groupby('Year')['Value'].sum().reset_index()
total_prod = total_prod[total_prod['Value'].notna() & (total_prod['Value'] > 0)]

ts = total_prod.set_index('Year')['Value']
ts = ts.interpolate(method='linear')
ts = ts.sort_index()

logger.info(f"Time series length: {len(ts)}")
logger.info(f"Date range: {ts.index.min()} to {ts.index.max()}")
logger.info(f"Value range: {ts.min():.2f} to {ts.max():.2f}")

# Visualize
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(ts.index, ts.values, linewidth=2, color='#1f77b4')
ax.set_title('Oklahoma Energy Production (1970-2023)', fontsize=14, fontweight='bold')
ax.set_ylabel('Energy Production', fontsize=11)
plt.tight_layout()
plt.savefig('energy_production_series.png', dpi=300, bbox_inches='tight')
plt.show()



# Code block 2
# Install TimesFM
# pip install timesfm

# Note: TimesFM requires specific dependencies
# Make sure you have PyTorch installed



# Code block 3
from timesfm import TimesFm
import numpy as np

# Initialize TimesFM model
# TimesFM is a zero-shot foundation model - no training needed!
model = TimesFm(
    context_len=512,  # Input context length (max historical data)
    horizon_len=128,  # Forecast horizon (max future predictions)
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend="cpu",  # or "gpu" for GPU acceleration
)

logger.info("TimesFM model loaded successfully")
logger.info(f"Context length: {model.context_len}")
logger.info(f"Horizon length: {model.horizon_len}")

# Prepare data for forecasting
# TimesFM expects numpy arrays
context_data = ts.values[-min(512, len(ts)):]  # Last 512 points (or all if shorter)
forecast_horizon = 24  # Forecast 24 periods ahead

logger.info(f"\nUsing {len(context_data)} points as context")
logger.info(f"Forecasting {forecast_horizon} periods ahead")

# Make zero-shot forecast
forecast = model.forecast(
    context=context_data,
    horizon=forecast_horizon
)

logger.info(f"\nForecast shape: {forecast.shape}")
logger.info(f"Forecast range: {forecast.min():.2f} to {forecast.max():.2f}")
logger.info(f"First 10 forecast values: {forecast[:10]}")



# Code block 4
# Simple forecasting example
import time

# Use last 100 points as context
context = ts.values[-100:]
horizon = 10

start_time = time.time()
forecast = model.forecast(context=context, horizon=horizon)
inference_time = time.time() - start_time

logger.info(f"Inference time: {inference_time:.3f} seconds")
logger.info(f"Forecast: {forecast}")

# Visualize
fig, ax = plt.subplots(figsize=(14, 6))

# Historical data
historical_dates = ts.index[-50:]
ax.plot(historical_dates, ts.values[-50:], 'b-', linewidth=2, label='Historical', alpha=0.7)

# Forecast
forecast_dates = pd.date_range(start=ts.index[-1] + pd.DateOffset(years=1), 
                               periods=horizon, freq='YS')
ax.plot(forecast_dates, forecast, 'r--', linewidth=2, label='TimesFM Forecast', marker='o')

ax.axvline(ts.index[-1], color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax.set_title('TimesFM Zero-Shot Forecast', fontsize=14, fontweight='bold')
ax.set_ylabel('Energy Production', fontsize=11)
ax.legend(frameon=True, fancybox=True, shadow=True)
plt.tight_layout()
plt.savefig('timesfm_forecast.png', dpi=300, bbox_inches='tight')
plt.show()



# Code block 5
from flask import Flask, request, jsonify
import numpy as np
from datetime import datetime, timedelta
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model instance (load once, reuse across requests)
logger.info("Loading TimesFM model...")
model = TimesFm(
    context_len=512,
    horizon_len=128,
    input_patch_len=32,
    output_patch_len=128,
    num_layers=20,
    model_dims=1280,
    backend="cpu",
)
logger.info("TimesFM model loaded successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        "status": "healthy",
        "model": "TimesFM",
        "context_len": model.context_len,
        "horizon_len": model.horizon_len
    }), 200

@app.route('/forecast', methods=['POST'])
def forecast():
    """
    Forecast endpoint
    
    Expected JSON:
    {
        "data": [1.2, 3.4, 5.6, ...],  # Time series values
        "horizon": 24,  # Forecast horizon (optional, default 24)
        "context_length": 512  # Optional, defaults to min(512, len(data))
    }
    
    Returns:
    {
        "forecast": [pred1, pred2, ...],
        "dates": ["2025-01-01", ...],
        "context_length": 512,
        "horizon": 24,
        "latency_ms": 123.45
    }
    """
    start_time = time.time()
    
    try:
        data = request.json
        if not data or 'data' not in data:
            return jsonify({"error": "Missing 'data' field"}), 400
        
        time_series = np.array(data['data'], dtype=np.float32)
        horizon = data.get('horizon', 24)
        
        # Validate input
        if len(time_series) < 32:
            return jsonify({"error": "Input too short (minimum 32 points)"}), 400
        
        if horizon < 1 or horizon > model.horizon_len:
            return jsonify({
                "error": f"Horizon must be between 1 and {model.horizon_len}"
            }), 400
        
        # Use last context_length points
        context_length = data.get('context_length', min(model.context_len, len(time_series)))
        context = time_series[-context_length:]
        
        # Make forecast
        forecast_values = model.forecast(
            context=context,
            horizon=horizon
        )
        
        # Generate timestamps (assuming daily data)
        last_date = datetime.now()
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
        
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        response = {
            "forecast": forecast_values.tolist(),
            "dates": [d.isoformat() for d in forecast_dates],
            "context_length": len(context),
            "horizon": horizon,
            "latency_ms": round(latency, 2)
        }
        
        logger.info(f"Forecast generated: horizon={horizon}, context={len(context)}, latency={latency:.2f}ms")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Forecast error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run in production with: gunicorn -w 4 -b 0.0.0.0:5000 app:app
    app.run(host='0.0.0.0', port=5000, debug=False)



# Code block 6
class TimesFMService:
    """Production service for batch forecasting"""
    
    def __init__(self, model_config=None):
        logger.info("Initializing TimesFM service...")
        self.model = TimesFm(
            context_len=512,
            horizon_len=128,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="cpu",
        )
        self.cache = {}  # Simple cache for repeated forecasts
        logger.info("TimesFM service initialized")
    
    def forecast_batch(self, time_series_list, horizon=24):
        """
        Forecast multiple time series in batch.
        
        Parameters:
        -----------
        time_series_list : list of arrays
            List of time series to forecast
        horizon : int
            Forecast horizon
        
        Returns:
        --------
        forecasts : ndarray
            Array of forecasts (n_series, horizon)
        """
        forecasts = []
        
        for i, ts in enumerate(time_series_list):
            ts_array = np.array(ts, dtype=np.float32)
            
            # Check cache (simple implementation)
            cache_key = (tuple(ts_array[-100:]), horizon)  # Cache key from last 100 points
            if cache_key in self.cache:
                forecasts.append(self.cache[cache_key])
                continue
            
            # Make forecast
            context = ts_array[-min(512, len(ts_array)):]
            forecast = self.model.forecast(context=context, horizon=horizon)
            
            # Cache result
            self.cache[cache_key] = forecast
            forecasts.append(forecast)
        
        return np.array(forecasts)
    
    def clear_cache(self):
        """Clear forecast cache"""
        self.cache.clear()
        logger.info("Cache cleared")

# Usage example
service = TimesFMService()

# Batch forecast multiple energy production series
multiple_series = [
    ts.values[-200:],  # Series 1
    ts.values[-300:],  # Series 2
    ts.values[-150:],  # Series 3
]

start_time = time.time()
batch_forecasts = service.forecast_batch(multiple_series, horizon=24)
batch_time = time.time() - start_time

logger.info(f"Batch forecasts shape: {batch_forecasts.shape}")
logger.info(f"Batch processing time: {batch_time:.3f} seconds")
logger.info(f"Average time per series: {batch_time/len(multiple_series):.3f} seconds")



# Code block 7
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncTimesFMService:
    """Async service for high-throughput forecasting"""
    
    def __init__(self, max_workers=4):
        logger.info(f"Initializing async TimesFM service with {max_workers} workers...")
        self.model = TimesFm(
            context_len=512,
            horizon_len=128,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
            backend="cpu",
        )
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info("Async TimesFM service initialized")
    
    async def forecast_async(self, context, horizon=24):
        """Async forecast"""
        loop = asyncio.get_event_loop()
        forecast = await loop.run_in_executor(
            self.executor,
            self.model.forecast,
            context,
            horizon
        )
        return forecast
    
    async def forecast_batch_async(self, contexts, horizon=24):
        """Batch async forecasts"""
        tasks = [self.forecast_async(ctx, horizon) for ctx in contexts]
        forecasts = await asyncio.gather(*tasks)
        return np.array(forecasts)

# Usage example
async_service = AsyncTimesFMService(max_workers=4)

# Async batch forecasting
async def main():
    contexts = [ts.values[-512:] for _ in range(10)]  # 10 series
    
    start_time = time.time()
    forecasts = await async_service.forecast_batch_async(contexts, horizon=24)
    elapsed = time.time() - start_time
    
    logger.info(f"Forecasted {len(contexts)} series in {elapsed:.2f} seconds")
    logger.info(f"Throughput: {len(contexts)/elapsed:.2f} forecasts/second")

# Run async
# asyncio.run(main())



# Code block 8
from collections import defaultdict
from dataclasses import dataclass, field

@dataclass
class ForecastMetrics:
    """Track forecasting performance metrics"""
    total_requests: int = 0
    successful_forecasts: int = 0
    failed_forecasts: int = 0
    avg_latency: float = 0.0
    latency_history: list = field(default_factory=list)
    
    def record_forecast(self, success: bool, latency: float):
        """Record forecast attempt"""
        self.total_requests += 1
        if success:
            self.successful_forecasts += 1
        else:
            self.failed_forecasts += 1
        
        self.latency_history.append(latency)
        # Keep only last 1000 for memory efficiency
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-1000:]
        
        self.avg_latency = np.mean(self.latency_history)
    
    def get_stats(self):
        """Get current statistics"""
        p95_latency = np.percentile(self.latency_history, 95) if self.latency_history else 0
        p99_latency = np.percentile(self.latency_history, 99) if self.latency_history else 0
        
        return {
            "total_requests": self.total_requests,
            "success_rate": self.successful_forecasts / max(self.total_requests, 1),
            "avg_latency_ms": self.avg_latency * 1000,
            "p95_latency_ms": p95_latency * 1000,
            "p99_latency_ms": p99_latency * 1000,
            "failed_count": self.failed_forecasts
        }

# Instrumented forecast function
metrics = ForecastMetrics()

def forecast_with_metrics(context, horizon=24):
    """Forecast with performance tracking"""
    start_time = time.time()
    try:
        forecast = model.forecast(context=context, horizon=horizon)
        latency = time.time() - start_time
        metrics.record_forecast(True, latency)
        return forecast
    except Exception as e:
        latency = time.time() - start_time
        metrics.record_forecast(False, latency)
        logger.error(f"Forecast failed: {e}")
        raise e

# Example usage
for _ in range(10):
    try:
        forecast_with_metrics(ts.values[-512:], horizon=24)
    except:
        pass

logger.info("Forecast Metrics:")
stats = metrics.get_stats()
for key, value in stats.items():
    logger.info(f"  {key}: {value}")



# Code block 9
# gunicorn -w 4 -b 0.0.0.0:5000 app:app



# Code block 10
# FROM python:3.10
WORKDIR /app
# COPY requirements.txt .
# RUN pip install -r requirements.txt
# COPY app.py .
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]



# Code block 11
# Complete code for reproducibility
# All imports, data loading, model setup, API, batch processing, and monitoring
# See individual code blocks above for full implementation



logger.info("All images generated successfully!")
