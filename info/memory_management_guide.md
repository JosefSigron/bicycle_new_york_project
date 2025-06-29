# Memory Management Guide for CitiBike Prediction

## Overview

Working with large datasets (6+ GB) can cause memory issues. This guide helps you choose the right approach for your system.

## System Requirements

### Minimum Requirements
- **RAM**: 4GB (for lightweight processing)
- **Storage**: 2GB free space
- **Time**: 10-30 minutes

### Recommended Requirements
- **RAM**: 8GB+ (for full dataset)
- **Storage**: 10GB free space
- **Time**: 1-2 hours for full processing

## Loading Options

### 🚀 Quick Start (Recommended)
```bash
python src/run_ride_prediction.py
```
- Automatically uses lightweight mode (500K rows)
- Fast processing (~5-10 minutes)
- Good for testing and development

### 🔧 Custom Processing
```bash
python src/run_full_prediction.py
```
- Interactive menu with 7 data loading options
- Choose based on your system capabilities
- Memory monitoring included

## Data Loading Strategies

| Option | Data Size | Memory Usage | Processing Time | Use Case |
|--------|-----------|--------------|-----------------|----------|
| Lightweight | 500K rows | ~200MB | 5-10 min | Quick testing |
| Single Year Sample | 10% of 2024 | ~400MB | 10-15 min | Development |
| Single Year Full | All 2024 | ~1.5GB | 30-45 min | Recent analysis |
| Recent Years Sample | 10% of 2022-2024 | ~800MB | 20-30 min | Trend analysis |
| Recent Years Full | All 2022-2024 | ~4GB | 60-90 min | Comprehensive |
| All Years Sample | 10% of all years | ~1.2GB | 30-45 min | Historical view |
| All Years Full | Complete dataset | ~6GB+ | 2+ hours | Research |

## Memory Optimization Features

### Chunked Processing
- Data loaded in 50K-100K row chunks
- Processes one chunk at a time
- Automatic garbage collection

### Smart Sampling
- Random sampling for representative data
- Maintains data distribution
- Configurable sample sizes

### Efficient Aggregation
- Streaming aggregation by time windows
- Memory cleanup after each step
- Progress monitoring

## Troubleshooting Memory Issues

### 1. Memory Error During Loading
**Problem**: "MemoryError" or system freezing
**Solutions**:
- Choose smaller data option (1-3)
- Close other applications
- Restart and try again

### 2. Slow Performance
**Problem**: System becomes very slow
**Solutions**:
- Use sample options (10% data)
- Reduce chunk size in code
- Monitor with Task Manager

### 3. Disk Space Issues
**Problem**: "No space left on device"
**Solutions**:
- Clean up temporary files
- Use smaller datasets
- Save to different drive

## Code Examples

### Loading Lightweight Data
```python
from ride_prediction_preprocessing import RideDataPreprocessor

preprocessor = RideDataPreprocessor()
data = preprocessor.load_data_lightweight(max_total_rows=500000)
```

### Loading with Sampling
```python
# Load 10% of 2024 data
data = preprocessor.load_data_single_year(year=2024, sample_fraction=0.1)

# Load 10% of multiple years
data = preprocessor.load_data_chunked(
    years=[2022, 2023, 2024], 
    sample_fraction=0.1
)
```

### Custom Chunk Size
```python
# Smaller chunks for limited memory
preprocessor = RideDataPreprocessor(chunk_size=50000)
```

## Monitoring Memory Usage

### Built-in Monitoring
The scripts automatically show:
- System RAM available
- Data file sizes
- Memory usage during processing
- Processing times

### Manual Monitoring
```python
import psutil

# Check available memory
memory = psutil.virtual_memory()
print(f"Available: {memory.available / 1024**3:.1f} GB")
print(f"Used: {memory.percent}%")
```

## Performance Tips

### 1. Close Other Applications
- Close web browsers, IDEs, etc.
- Free up as much RAM as possible

### 2. Use SSD if Available
- Faster disk I/O improves performance
- Temporary files benefit from SSD

### 3. Process During Off-Peak
- Run large jobs overnight
- Avoid competing with other processes

### 4. Incremental Processing
- Process one year at a time
- Combine results later if needed

### 5. Save Processed Data
- Reuse aggregated data
- Skip preprocessing on subsequent runs

## Hardware Recommendations

### For Development/Testing
- 8GB RAM
- SSD storage
- Multi-core CPU

### For Production/Research
- 16GB+ RAM
- NVMe SSD
- 8+ core CPU
- GPU (optional, for deep learning)

## Error Recovery

### If Process Crashes
1. Check available memory
2. Restart Python/system
3. Try smaller dataset
4. Use processed data if available

### If Results Look Wrong
1. Check data sample size
2. Verify aggregation logic
3. Compare with full dataset results
4. Review preprocessing steps

## Advanced Options

### Using Dask (Optional)
For very large datasets, consider Dask:
```python
import dask.dataframe as dd

# Read with Dask
df = dd.read_parquet("data/combined/*.parquet")
result = df.groupby("hour").agg({"trip_duration": "count"}).compute()
```

### Custom Aggregation
For specific use cases:
```python
# Custom aggregation function
def custom_agg(df):
    return df.groupby(['hour', 'weather_cat']).agg({
        'trip_duration': ['count', 'mean']
    })

# Apply to chunks
results = []
for chunk in pd.read_parquet(file, chunksize=100000):
    results.append(custom_agg(chunk))
```

---

*For technical support, check the main documentation or create an issue.* 