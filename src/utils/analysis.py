from openai import OpenAI
import json
import yaml
import re
import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64
import io
from collections import Counter
import tiktoken


def extract_json_from_response(text: str) -> str:
    """Extract JSON from a response that might contain markdown code blocks."""
    # Try to find JSON within code blocks first
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # If no code blocks, try to find raw JSON
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    # If no JSON found, return the original text
    return text


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text using tiktoken."""
    try:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(str(text)))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0


def create_distribution_plot(data, column):
    """Create a distribution plot using Plotly and convert to image."""
    try:
        # Check if the column contains lists
        if isinstance(data[column].iloc[0], list):
            print(f"Processing list column: {column}")
            value_counts = flatten_list_column(data, column)

            fig = go.Figure(
                [
                    go.Bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        marker=dict(
                            color=value_counts.values,
                            colorscale=px.colors.sequential.Plotly3,
                        ),
                    )
                ]
            )

        else:
            if data[column].dtype in ["int64", "float64"]:
                # Continuous data - use histogram
                fig = go.Figure()
                fig.add_trace(
                    go.Histogram(
                        x=data[column],
                        name="Count",
                        nbinsx=30,
                        marker=dict(
                            color="rgba(110, 68, 255, 0.7)",
                            line=dict(color="rgba(184, 146, 255, 1)", width=1),
                        ),
                    )
                )
            else:
                # Categorical data
                value_counts = data[column].value_counts()
                fig = go.Figure(
                    [
                        go.Bar(
                            x=value_counts.index,
                            y=value_counts.values,
                            marker=dict(
                                color=value_counts.values,
                                colorscale=px.colors.sequential.Plotly3,
                            ),
                        )
                    ]
                )

        # Common layout updates
        fig.update_layout(
            title=dict(text=f"Distribution of {column}", x=0.5, y=0.95),
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white",
            margin=dict(t=50, l=50, r=30, b=50),
            width=600,
            height=400,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )

        # Rotate x-axis labels if needed
        if isinstance(data[column].iloc[0], list) or data[column].dtype not in [
            "int64",
            "float64",
        ]:
            fig.update_layout(xaxis_tickangle=-45)

        # Update grid style
        fig.update_yaxes(gridcolor="rgba(128,128,128,0.1)", gridwidth=1)
        fig.update_xaxes(gridcolor="rgba(128,128,128,0.1)", gridwidth=1)

        # Convert to PNG with moderate resolution
        img_bytes = fig.to_image(format="png", scale=1.5)

        # Encode to base64
        img_base64 = base64.b64encode(img_bytes).decode()

        return img_base64

    except Exception as e:
        print(f"Error creating distribution plot for {column}: {str(e)}")
        raise e


def create_wordcloud(data, column):
    """Create a word cloud visualization."""
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    try:
        # Handle list columns
        if isinstance(data[column].iloc[0], list):
            text = " ".join(
                [
                    " ".join(map(str, sublist))
                    for sublist in data[column]
                    if isinstance(sublist, list)
                ]
            )
        else:
            # Handle regular columns
            text = " ".join(data[column].astype(str))

        wordcloud = WordCloud(
            width=600,
            height=300,
            background_color="white",
            colormap="plasma",
            max_words=100,
        ).generate(text)

        # Create matplotlib figure
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {column}")

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close()
        buf.seek(0)

        # Convert to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode()

        return img_base64

    except Exception as e:
        print(f"Error creating word cloud for {column}: {str(e)}")
        raise e


def analyze_dataset_with_openai(client: OpenAI, data) -> dict:
    """Analyze dataset using OpenAI API with improved type inference and efficient sampling."""
    # Convert dictionary to DataFrame if needed
    if isinstance(data, dict):
        df = pd.DataFrame(data)
    else:
        df = data

    # Take a very small sample for efficiency
    sample_size = min(3, len(df))
    if len(df) > 3:
        sample_indices = df.index[:sample_size]  # Take first 3 rows instead of random sampling
        sample_df = df.loc[sample_indices]
    else:
        sample_df = df
        
    dataset_sample = sample_df.to_dict('records')
    single_record = dataset_sample[0]

    # Create type hints dictionary - only process the sample
    type_hints = {}
    for column in sample_df.columns:
        # Get the pandas dtype
        dtype = sample_df[column].dtype
        
        # Efficiently identify types without complex operations
        if pd.api.types.is_integer_dtype(dtype):
            type_hints[column] = "integer"
        elif pd.api.types.is_float_dtype(dtype):
            type_hints[column] = "number"
        elif pd.api.types.is_bool_dtype(dtype):
            type_hints[column] = "boolean"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            type_hints[column] = "datetime"
        elif pd.api.types.is_categorical_dtype(dtype):
            type_hints[column] = "categorical"
        elif pd.api.types.is_string_dtype(dtype):
            # Simple check for list-like values
            first_val = sample_df[column].iloc[0]
            if isinstance(first_val, list):
                type_hints[column] = "array"
            else:
                type_hints[column] = "string"
        else:
            type_hints[column] = "unknown"

    prompt = f"""Analyze this dataset sample and provide the following in a JSON response:

    1. A concise description that includes:
       - A one-sentence overview of what the dataset contains
       - A bullet-pointed list of key features and statistics
       - A brief statement about potential ML/AI applications
    
    2. A schema showing each field's type and description. Here is the actual DataFrame type information:
    {json.dumps(type_hints, indent=2)}
    
    And here's a single record for reference:
    {json.dumps(single_record, indent=2)}
    
    3. A formatted example record

    Format your response as JSON with these exact keys:

    {{
        "description": {{
            "overview": "One clear sentence describing the dataset...",
            "key_features": [
                "Feature or statistic 1",
                "Feature or statistic 2"
            ],
            "ml_applications": "Brief statement about ML/AI use cases..."
        }},
        "schema": {{
            "field_name": {{
                "type": "use the type from the provided type_hints",
                "description": "Description of what this field contains"
            }}
        }},
        "example": {{"key": "value"}}
    }}
    
    For context, here are more sample records:
    {json.dumps(dataset_sample, indent=2)}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )

        # Get the response content
        response_text = response.choices[0].message.content
        
        # Extract JSON from the response
        json_str = extract_json_from_response(response_text)
        
        # Parse the JSON
        result = json.loads(json_str)
        return result

    except Exception as e:
        print(f"OpenAI API error: {str(e)}")
        return {
            "description": {
                "overview": "Error analyzing dataset",
                "key_features": ["Error: Failed to analyze dataset"],
                "ml_applications": "Analysis unavailable",
            },
            "schema": {},
            "example": {},
        }

def analyze_dataset_statistics(df):
    """Generate simplified dataset statistics with token counting."""
    stats = {
        "basic_stats": {
            "total_records": len(df),
            "total_features": len(df.columns),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB"
        },
        "token_stats": {
            "total": 0,
            "by_column": {}
        }
    }
    
    # Count tokens for each column
    for column in df.columns:
        try:
            if df[column].dtype == 'object' or isinstance(df[column].iloc[0], list):
                # For list columns, join items into strings
                if isinstance(df[column].iloc[0], list):
                    token_counts = df[column].apply(lambda x: count_tokens(' '.join(str(item) for item in x)))
                else:
                    token_counts = df[column].apply(lambda x: count_tokens(str(x)))
                
                total_tokens = int(token_counts.sum())
                stats["token_stats"]["total"] += total_tokens
                stats["token_stats"]["by_column"][column] = total_tokens
        except Exception as e:
            print(f"Error processing column {column}: {str(e)}")
            continue
    
    return stats

def format_dataset_stats(stats):
    """Format simplified dataset statistics as markdown."""
    md = """## Dataset Overview

### Basic Statistics
* Total Records: {total_records:,}
* Total Features: {total_features}
* Memory Usage: {memory_usage}
""".format(**stats["basic_stats"])

    # Token Statistics
    if stats["token_stats"]["total"] > 0:
        md += "\n### Token Info\n"
        md += f"* Total Tokens: {stats['token_stats']['total']:,}\n"
        if stats["token_stats"]["by_column"]:
            md += "\nTokens by Column:\n"
            for col, count in stats["token_stats"]["by_column"].items():
                md += f"* {col}: {count:,}\n"

    return md

def generate_dataset_card(
    dataset_info: dict,
    distribution_plots: dict,
    wordcloud_plots: dict,
    openai_analysis: dict,
    df: pd.DataFrame,
) -> str:
    """Generate the complete dataset card content."""
    yaml_content = {
        "language": ["en"],
        "license": "apache-2.0",
        "multilinguality": "monolingual",
        "size_categories": ["1K<n<10K"],
        "task_categories": ["other"],
    }

    yaml_string = yaml.dump(yaml_content, sort_keys=False)
    description = openai_analysis["description"]

    # Generate schema table
    schema_table = generate_schema_table(openai_analysis["schema"])

    # Format example as JSON code block
    example_block = f"```json\n{json.dumps(openai_analysis['example'], indent=2)}\n```"

    # Generate dataset statistics
    stats = analyze_dataset_statistics(df)
    stats_section = format_dataset_stats(stats)

    # Add distribution plots inline
    distribution_plots_md = ""
    if distribution_plots:
        distribution_plots_md = "\n### Distribution Plots\n\n"
        distribution_plots_md += '<div style="display: grid; grid-template-columns: repeat(1, 1fr); gap: 20px;">\n'
        for col, img_str in distribution_plots.items():
            distribution_plots_md += f"<div>\n"
            distribution_plots_md += f"<h4>Distribution of {col}</h4>\n"
            distribution_plots_md += f'<img src="data:image/png;base64,{img_str}" style="width: 100%; height: auto;">\n'
            distribution_plots_md += "</div>\n"
        distribution_plots_md += "</div>\n\n"

    # Add word clouds inline in a grid
    wordcloud_plots_md = ""
    if wordcloud_plots:
        wordcloud_plots_md = "\n### Word Clouds\n\n"
        wordcloud_plots_md += '<div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">\n'
        for col, img_str in wordcloud_plots.items():
            wordcloud_plots_md += f"<div>\n"
            wordcloud_plots_md += f"<h4>Word Cloud for {col}</h4>\n"
            wordcloud_plots_md += f'<img src="data:image/png;base64,{img_str}" style="width: 100%; height: auto;">\n'
            wordcloud_plots_md += "</div>\n"
        wordcloud_plots_md += "</div>\n\n"

    # Generate clean dataset name for citation
    clean_dataset_name = dataset_info["dataset_name"].replace("/", "_")

    # Build the markdown content
    readme_content = f"""---
{yaml_string}---

# {dataset_info['dataset_name']}

{description['overview']}

The dataset includes:
{chr(10).join(f'* {feature}' for feature in description['key_features'])}

{description['ml_applications']}

## Dataset Schema

{schema_table}

## Example Record

{example_block}

## Data Distribution Analysis

The following visualizations show key characteristics of the dataset:

{distribution_plots_md}
{wordcloud_plots_md}

{stats_section}

## Citation and Usage

If you use this dataset in your research or applications, please cite it as:

```bibtex
@dataset{{{clean_dataset_name},
    title = {{{dataset_info['dataset_name']}}},
    author = {{Dataset Authors}},
    year = {{{datetime.datetime.now().year}}},
    publisher = {{Hugging Face}},
    howpublished = {{Hugging Face Datasets}},
    url = {{https://huggingface.co/datasets/{dataset_info['dataset_name']}}}
}}
```

### Usage Guidelines

This dataset is released under the Apache 2.0 License. When using this dataset:

* 📚 Cite the dataset using the BibTeX entry above
* 🤝 Consider contributing improvements or reporting issues
* 💡 Share derivative works with the community when possible

For questions or additional information, please visit the dataset repository on Hugging Face.
"""

    return readme_content


def generate_schema_table(schema: dict) -> str:
    """Generate a markdown table for the schema, handling nested structures."""
    # Table header
    table = "| Field | Type | Description |\n| --- | --- | --- |\n"

    # Generate rows recursively
    rows = []
    for field, info in schema.items():
        rows.extend(format_schema_item(field, info))

    # Join all rows
    table += "\n".join(rows)
    return table


def format_schema_item(field_name: str, field_info: dict, prefix: str = "") -> list:
    """Recursively format schema items for nested structures."""
    rows = []

    # Handle nested objects
    if isinstance(field_info, dict):
        if "type" in field_info and "description" in field_info:
            # This is a leaf node with type and description
            rows.append(
                f"| {prefix}{field_name} | {field_info['type']} | {field_info['description']} |"
            )
        else:
            # This is a nested object, recurse through its properties
            for subfield, subinfo in field_info.items():
                if prefix:
                    new_prefix = f"{prefix}{field_name}."
                else:
                    new_prefix = f"{field_name}."
                rows.extend(format_schema_item(subfield, subinfo, new_prefix))

    return rows


def flatten_list_column(data, column):
    """Flatten a column containing lists into individual values with counts."""
    # Flatten the lists into individual items
    flattened = [
        item
        for sublist in data[column]
        if isinstance(sublist, list)
        for item in sublist
    ]
    # Count occurrences
    value_counts = pd.Series(Counter(flattened))
    return value_counts
