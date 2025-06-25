# Audio Modality Analyzer

## Implementation Overview
Analyzer for audio datasets from HuggingFace Hub. Focuses on basic audio characteristics without heavy speech processing.

## Core Audio Analysis

### Audio Properties
- Sample rate, duration, and channel configuration
- File format and codec detection
- Basic amplitude and frequency analysis
- Audio quality metrics (simple SNR estimation)

### Content Analysis (Basic)
- Simple voice activity detection
- Basic audio classification (speech vs music vs silence)
- Volume level analysis

## Format Support
WAV, MP3, FLAC, OGG with automatic format detection.

## Analysis Configurations

### Quick Mode
- Sample size: 30 audio files
- Basic properties only
- Simple waveform analysis

### Standard Mode
- Sample size: 200 audio files
- Full audio characteristics
- Basic content classification
- Quality assessment

### Comprehensive Mode
- Sample size: 1,000 audio files
- Advanced audio metrics
- Detailed spectral analysis

## Audio Processing
- Uses librosa for basic audio analysis
- Simple spectral features (MFCCs for classification)
- Avoids heavy speech recognition models
- Focus on audio signal properties

## Visualization
- Waveform plots
- Duration distributions
- Sample rate analysis
- Simple spectrograms

## Implementation Notes
- Lightweight audio processing without AI models
- Memory-efficient loading with duration limits
- Focus on technical audio properties rather than content understanding 