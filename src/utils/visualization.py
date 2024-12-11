import base64
import io
from collections import Counter

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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


def create_distribution_plot(data, column):
    """Create a beautiful distribution plot using Plotly and convert to image."""
    try:
        # Check if the column contains lists
        if isinstance(data[column].iloc[0], list):
            print(f"Processing list column: {column}")
            value_counts = flatten_list_column(data, column)
        else:
            # Handle regular columns
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

            else:
                # Categorical data
                value_counts = data[column].value_counts()

        # For both list columns and categorical data
        if "value_counts" in locals():
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
            title=f"Distribution of {column}",
            xaxis_title=column,
            yaxis_title="Count",
            template="plotly_white",
            margin=dict(t=50, l=50, r=50, b=50),
            width=1200,
            height=800,
            showlegend=False,
        )

        # Rotate x-axis labels if needed
        if isinstance(data[column].iloc[0], list) or data[column].dtype not in [
            "int64",
            "float64",
        ]:
            fig.update_layout(xaxis_tickangle=-45)

        # Convert to PNG
        img_bytes = fig.to_image(format="png", scale=2.0)

        # Encode to base64
        img_base64 = base64.b64encode(img_bytes).decode()

        return img_base64

    except Exception as e:
        print(f"Error creating distribution plot for {column}: {str(e)}")
        raise e


def create_wordcloud(data, column):
    """Create a word cloud visualization."""
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

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
            width=1200,
            height=800,
            background_color="white",
            colormap="plasma",
            max_words=100,
        ).generate(text)

        # Create matplotlib figure
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(f"Word Cloud for {column}")

        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
        plt.close()
        buf.seek(0)

        # Convert to base64
        img_base64 = base64.b64encode(buf.getvalue()).decode()

        return img_base64

    except Exception as e:
        print(f"Error creating word cloud for {column}: {str(e)}")
        raise e


def create_wordcloud(data, column):
    """Create a word cloud visualization."""
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    # Generate word cloud
    text = " ".join(data[column].astype(str))
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="plasma",
        max_words=100,
    ).generate(text)

    # Create matplotlib figure
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Word Cloud for {column}")

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=300)
    plt.close()
    buf.seek(0)

    # Convert to base64
    img_base64 = base64.b64encode(buf.getvalue()).decode()

    return img_base64
