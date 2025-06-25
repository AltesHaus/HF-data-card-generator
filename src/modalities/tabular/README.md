# Tabular Modality Analyzer

## Implementation Overview
Analyzer for structured datasets (CSV, JSON, Parquet) from HuggingFace Hub. Focuses on basic statistical analysis and data profiling.

## Core Analysis Functions

### Statistical Processing
- Descriptive statistics for numerical columns (mean, median, std, quartiles)
- Missing value patterns and data completeness
- Basic correlation analysis for numerical features
- Simple outlier detection using IQR method

### Data Profiling
- Column type inference (numerical, categorical, text, datetime)
- Categorical value distributions and cardinality
- Basic data quality assessment
- Sample preview generation

## Format Support
CSV/TSV, JSON/JSONL, Parquet with automatic delimiter and encoding detection.

## Analysis Configurations

### Quick Mode
- Sample size: 500 rows
- Basic statistics only
- Simple visualizations (histograms, bar charts)

### Standard Mode
- Sample size: 5,000 rows
- Full statistical profiling
- Correlation analysis
- Data quality assessment

### Comprehensive Mode
- Sample size: 20,000 rows (or full dataset if smaller)
- Advanced statistics
- Bias detection (basic)

## Visualization
- Distribution plots for numerical columns
- Bar charts for categorical data
- Missing data patterns
- Simple correlation heatmaps

## Implementation Notes
- Uses pandas for data processing
- Focuses on memory-efficient operations
- Graceful handling of large datasets through sampling