# Image Modality Analyzer

## Implementation Overview
Analyzer for image datasets from HuggingFace Hub. Focuses on basic image properties and simple computer vision.

## Core Analysis Functions

### Image Properties
- Dimensions, format, and file size analysis
- Color mode detection (RGB, grayscale, etc.)
- Basic color statistics (brightness, contrast)
- Compression and quality assessment

### Simple Computer Vision
- Basic image quality metrics (sharpness using Laplacian)
- Duplicate detection using perceptual hashing
- Simple color palette extraction

## Format Support
JPEG, PNG, WebP, TIFF, BMP, GIF with robust loading.

## Analysis Configurations

### Quick Mode
- Sample size: 50 images
- Basic metadata extraction
- Simple statistics

### Standard Mode
- Sample size: 500 images
- Full property analysis
- Quality assessment
- Duplicate detection

### Comprehensive Mode
- Sample size: 2,000 images
- Advanced quality metrics
- Detailed color analysis

## Image Quality Assessment
- Sharpness detection using edge analysis
- Basic noise estimation
- Compression artifact detection
- Format consistency checking

## Visualization
- Dimension scatter plots
- Color histograms
- Quality metric distributions
- Sample image grids

## Implementation Notes
- Uses PIL/Pillow for image processing
- Avoids heavy deep learning models
- Focus on statistical image analysis
- Memory-efficient processing with resizing for analysis 