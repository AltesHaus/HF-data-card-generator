# Text Modality Analyzer

## Implementation Overview
Analyzer for text datasets from HuggingFace Hub. Focuses on basic linguistic analysis without heavy NLP models.

## Core Analysis Functions

### Basic Text Processing
- Language detection using lightweight methods
- Text statistics (character, word, sentence counts)
- Length distribution analysis
- Basic readability scores (Flesch-Kincaid)

### Content Analysis
- Simple keyword extraction (TF-IDF)
- Basic duplicate detection using text similarity
- Encoding detection and validation

## Format Support
Plain text, JSON/JSONL (text fields), CSV (text columns), basic HTML/Markdown.

## Analysis Configurations

### Quick Mode
- Sample size: 200 texts
- Basic statistics and language detection
- Simple visualizations

### Standard Mode
- Sample size: 2,000 texts
- Full text analysis
- Keyword extraction
- Quality assessment

### Comprehensive Mode
- Sample size: 10,000 texts
- Advanced text metrics
- Cross-text similarity (basic)

## Text Quality Assessment
- Basic grammar checking (simple rules)
- Encoding consistency
- Length outlier detection
- Duplicate content identification

## Visualization
- Text length distributions
- Word frequency charts
- Language distribution
- Quality score distributions

## Implementation Notes
- Avoids heavy transformer models for cost efficiency
- Uses lightweight libraries (spaCy small models, NLTK basics)
- Focus on statistical rather than semantic analysis 