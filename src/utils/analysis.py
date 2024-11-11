import base64
import datetime
import io
import json
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import yaml
from openai import OpenAI


def create_distribution_plot(data, column):
    """Create a beautiful distribution plot using Plotly."""
    if data[column].dtype in ["int64", "float64"]:
        # Continuous data - use histogram
        fig = go.Figure()

        # Add histogram
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

        # Update layout
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white",
            showlegend=False,
            margin=dict(t=50, l=50, r=50, b=50),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            width=800,
            height=500,
        )

    else:
        # Categorical data - use bar plot
        value_counts = data[column].value_counts()

        fig = go.Figure(
            [
                go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    marker=dict(
                        color=value_counts.values,
                        colorscale=px.colors.sequential.Plotly3,
                        colorbar=dict(title="Count"),
                    ),
                )
            ]
        )

        # Update layout
        fig.update_layout(
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white",
            margin=dict(t=50, l=50, r=50, b=50),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            width=800,
            height=500,
        )

        # Rotate x-axis labels if needed
        if max([len(str(x)) for x in value_counts.index]) > 10:
            fig.update_layout(xaxis_tickangle=-45)

    # Convert to PNG
    img_bytes = fig.to_image(format="png", scale=2)
    img_str = base64.b64encode(img_bytes).decode()

    return img_str


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


def generate_dataset_card(
    dataset_info: dict,
    distribution_plots: dict,
    wordcloud_plots: dict,
    openai_analysis: dict,
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


def analyze_dataset_with_openai(client: OpenAI, dataset_sample) -> dict:
    """Analyze dataset sample using OpenAI API."""
    sample_json = json.dumps(dataset_sample, indent=2)

    prompt = f"""Analyze this dataset sample and provide the following in a JSON response:

    1. A concise description that includes:
       - A one-sentence overview of what the dataset contains
       - A bullet-pointed list of key features and statistics
       - A brief statement about potential ML/AI applications
    
    2. A schema where each field has a clear description of its contents and data type
    
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
                "type": "data type (string, number, etc.)",
                "description": "Description of what this field contains"
            }}
        }},
        "example": {{"key": "value"}}
    }}
    
    Sample data:
    {sample_json}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )

        # Get the response content
        response_text = response.choices[0].message.content
        print("OpenAI Response:", response_text)  # Debug print

        # Extract JSON from the response
        json_str = extract_json_from_response(response_text)
        print("Extracted JSON:", json_str)  # Debug print

        # Parse the JSON
        result = json.loads(json_str)
        print("Parsed Result:", result)  # Debug print
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
