import json

import pandas as pd
import streamlit as st
from datasets import load_dataset
from huggingface_hub import HfApi, login
from openai import OpenAI

# Import our utility functions
from utils.analysis import analyze_dataset_with_openai, generate_dataset_card
from utils.visualization import create_distribution_plot, create_wordcloud

# Initialize session state variables
if "openai_analysis" not in st.session_state:
    st.session_state.openai_analysis = None
if "df" not in st.session_state:
    st.session_state.df = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None
if "selected_dist_columns" not in st.session_state:
    st.session_state.selected_dist_columns = []
if "selected_wordcloud_columns" not in st.session_state:
    st.session_state.selected_wordcloud_columns = []

st.set_page_config(
    page_title="Dataset Card Generator",
    page_icon="📊",
    layout="wide",
)


def initialize_openai_client(api_key):
    """Initialize OpenAI client with API key."""
    return OpenAI(api_key=api_key)


def load_and_analyze_dataset(dataset_name):
    """Load dataset and perform initial analysis."""
    progress_container = st.empty()

    with progress_container.container():
        with st.status("Loading dataset...", expanded=True) as status:
            try:
                # Load dataset
                status.write("📥 Loading dataset from HuggingFace...")
                dataset = load_dataset(dataset_name, split="train")
                df = pd.DataFrame(dataset)
                st.session_state.df = df
                st.session_state.dataset_name = dataset_name

                # Initialize OpenAI analysis
                try:
                    status.write("🤖 Analyzing dataset ...")
                    client = initialize_openai_client(st.session_state.openai_key)
                    sample_data = dataset[:5]
                    print("Sample data:", json.dumps(sample_data, indent=2))

                    analysis = analyze_dataset_with_openai(client, sample_data)
                    print("Analysis result:", json.dumps(analysis, indent=2))

                    st.session_state.openai_analysis = analysis
                except Exception as e:
                    print(f"Analysis error: {str(e)}")
                    status.update(label=f"❌ Error: {str(e)}", state="error")

                status.update(
                    label="✅ Dataset loaded and analyzed successfully!",
                    state="complete",
                )

            except Exception as e:
                status.update(label=f"❌ Error: {str(e)}", state="error")
                st.error(f"Failed to load dataset: {str(e)}")
                return


def display_dataset_analysis():
    """Display dataset analysis and visualization options."""
    if st.session_state.df is None:
        return

    st.header("Dataset Analysis")

    # Dataset preview
    with st.expander("📊 Dataset Preview", expanded=True):
        st.dataframe(st.session_state.df.head(), use_container_width=True)

    # Column selection for visualizations
    st.subheader("Select Visualization Fields")

    col1, col2 = st.columns(2)

    with col1:
        # Distribution plot selection
        st.session_state.selected_dist_columns = st.multiselect(
            "Distribution Plots (max 2)",
            options=st.session_state.df.columns.tolist(),
            format_func=lambda x: get_column_type_description(st.session_state.df, x),
            max_selections=2,
            help="Select columns to show value distributions. List columns will show frequency of individual items.",
        )

    with col2:
        # Word cloud selection
        text_columns = [
            col
            for col in st.session_state.df.columns
            if st.session_state.df[col].dtype == "object"
            or isinstance(st.session_state.df[col].iloc[0], list)
        ]

        st.session_state.selected_wordcloud_columns = st.multiselect(
            "Word Clouds (max 2)",
            options=text_columns,
            format_func=lambda x: get_column_type_description(st.session_state.df, x),
            max_selections=2,
            help="Select text columns to generate word clouds",
        )

    # Add some spacing
    st.markdown("---")

    # Generate card button
    if st.button("Generate Dataset Card", type="primary", use_container_width=True):
        if not (
            st.session_state.selected_dist_columns
            or st.session_state.selected_wordcloud_columns
        ):
            st.warning(
                "Please select at least one visualization before generating the card."
            )
            return

        generate_and_display_card()


def generate_and_display_card():
    """Generate and display the dataset card with visualizations."""
    if not st.session_state.openai_analysis:
        st.error(
            "Dataset analysis not available. Please try loading the dataset again."
        )
        return

    with st.status("Generating dataset card...", expanded=True) as status:
        try:
            # Create visualizations
            status.write("📊 Creating distribution plots...")
            distribution_plots = {}
            for col in st.session_state.selected_dist_columns:
                print(f"Generating distribution plot for {col}")
                img_base64 = create_distribution_plot(st.session_state.df, col)
                distribution_plots[col] = img_base64
                print(f"Successfully created plot for {col}")

            status.write("🔤 Generating word clouds...")
            wordcloud_plots = {}
            for col in st.session_state.selected_wordcloud_columns:
                print(f"Generating word cloud for {col}")
                img_base64 = create_wordcloud(st.session_state.df, col)
                wordcloud_plots[col] = img_base64
                print(f"Successfully created word cloud for {col}")

            # Generate dataset card content
            status.write("📝 Composing dataset card...")
            dataset_info = {"dataset_name": st.session_state.dataset_name}

            readme_content = generate_dataset_card(
                dataset_info=dataset_info,
                distribution_plots=distribution_plots,
                wordcloud_plots=wordcloud_plots,
                openai_analysis=st.session_state.openai_analysis,
                df=st.session_state.df,  # Added DataFrame parameter
            )

            # Display results
            status.update(label="✅ Dataset card generated!", state="complete")

            # Display the markdown with images
            st.markdown(readme_content, unsafe_allow_html=True)

            # Add download button
            st.download_button(
                label="⬇️ Download Dataset Card",
                data=readme_content,
                file_name="README.md",
                mime="text/markdown",
                use_container_width=True,
            )

        except Exception as e:
            print(f"Error in generate_and_display_card: {str(e)}")
            st.error(f"Error generating dataset card: {str(e)}")
            raise e


def get_column_type_description(data, column):
    """Get a user-friendly description of the column type."""
    try:
        if isinstance(data[column].iloc[0], list):
            return f"{column} (list)"
        elif data[column].dtype in ["int64", "float64"]:
            return f"{column} (numeric)"
        else:
            return f"{column} (text/categorical)"
    except:
        return f"{column} (unknown)"


def get_api_keys():
    """Get API keys from secrets or user input."""
    # Try to get from secrets first
    try:
        hf_token = st.secrets["api_keys"]["huggingface"]
        openai_key = st.secrets["api_keys"]["openai"]
        return hf_token, openai_key
    except:
        return None, None


def get_secrets():
    """Get API keys from secrets.toml if it exists."""
    try:
        hf_token = st.secrets.get("api_keys", {}).get("huggingface", "")
        openai_key = st.secrets.get("api_keys", {}).get("openai", "")
        return hf_token, openai_key
    except Exception as e:
        print(f"No secrets file found or error reading secrets: {e}")
        return "", ""


def main():
    st.title("📊 Dataset Card Generator")
    st.markdown(
        """
    Generate beautiful documentation for your HuggingFace datasets with automated analysis, 
    visualizations, and formatted dataset cards.
    """
    )

    # Get secrets if available
    default_hf_token, default_openai_key = get_api_keys()

    # Authentication section in sidebar
    with st.sidebar:
        st.header("🔑 Authentication")

        # OpenAI API key (required)
        openai_key = st.text_input(
            "OpenAI API Key",
            value=default_openai_key,
            type="password" if not default_openai_key else "default",
            help="Required: Your OpenAI API key for dataset analysis",
        )

        # HuggingFace token (optional)
        hf_token = st.text_input(
            "HuggingFace Token (optional)",
            value=default_hf_token,
            type="password" if not default_hf_token else "default",
            help="Optional: Only required for private datasets",
        )

        if openai_key:
            try:
                # Only attempt HF login if token is provided
                if hf_token:
                    login(hf_token)
                    st.success("✅ HuggingFace authentication successful!")

                st.session_state.openai_key = openai_key
                st.success("✅ OpenAI API key set!")
            except Exception as e:
                st.error(f"❌ Authentication error: {str(e)}")
                return
        else:
            st.info("👆 Please enter your OpenAI API key to get started.")
            return

    # Main content area
    if not openai_key:
        return

    dataset_name = st.text_input(
        "Enter HuggingFace Dataset Name",
        placeholder="username/dataset",
        help="Enter the full path to your HuggingFace dataset (e.g., 'username/dataset')",
    )

    if dataset_name:
        if st.button("Load Dataset", type="primary"):
            load_and_analyze_dataset(dataset_name)

    if st.session_state.df is not None:
        display_dataset_analysis()


if __name__ == "__main__":
    main()
