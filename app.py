import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import os
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate AI insights
def generate_insights(df):
    # Compute descriptive statistics for numeric columns
    numeric_summary = df.describe().to_string()

    # Compute correlation matrix for numerical insights
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns

    correlation_matrix = numeric_df.corr().to_string()

    # Compute unique value counts for categorical columns
    categorical_summary = "\n".join(
        [f"{col}: {df[col].nunique()} unique values" for col in df.select_dtypes(include=["object"]).columns]
    )

    prompt = f"""Analyze the following dataset and provide key insights:
    - Identify trends and patterns.
    - Detect any anomalies or outliers.
    - Suggest suitable visualizations for better understanding.

    **Numerical Data Summary:**
    {numeric_summary}

    **Correlation Matrix:**
    {correlation_matrix}

    **Categorical Data Summary:**
    {categorical_summary}
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a data analyst."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error generating insights: {e}"


# Streamlit UI
st.title("📊 Automated Data Visualization & AI Insights")

@st.cache_data
def load_data(uploaded_file):
    try:
        # Load CSV as strings to avoid type inference issues
        df = pd.read_csv(uploaded_file, dtype=str)

        # Drop identifier columns like Retailer ID and Invoice Date
        id_columns = [col for col in df.columns if "id" in col.lower() or "date" in col.lower()]
        df = df.drop(columns=id_columns, errors="ignore")

        # Clean numeric columns (remove $ signs, commas, and convert to float)
        for col in df.columns:
            if df[col].str.contains(r"[\d,]", na=False).all():  # Check if column contains numbers
                df[col] = df[col].str.replace(r"[$,]", "", regex=True)  # Remove $ and , 
                df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to float/int

        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None




def chatbot_mode():
    st.subheader("💬 Chatbot Mode")
    
    uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.write("### Data Preview")
            st.write(df.head())

            # Initial Descriptive Statistics
            st.write("### Descriptive Statistics")
            st.write(df.describe())

            numeric_columns = df.select_dtypes(include=[np.number]).columns

            # Univariate Analysis
            st.write("### 📊 Univariate Analysis")
            for column in numeric_columns:
                st.write(f"#### {column}")
                try:
                    st.plotly_chart(px.histogram(df, x=column, title=f"Histogram of {column}"))
                    st.plotly_chart(px.box(df, y=column, title=f"Boxplot of {column}"))
                except Exception as e:
                    st.warning(f"Could not generate plots for {column}: {e}")

            # Bivariate Analysis
            st.write("### 🔗 Bivariate Analysis")
            if len(numeric_columns) >= 2:
                for i in range(len(numeric_columns)):
                    for j in range(i + 1, len(numeric_columns)):
                        x_col, y_col = numeric_columns[i], numeric_columns[j]
                        st.write(f"#### {x_col} vs {y_col}")
                        try:
                            st.plotly_chart(px.scatter(df, x=x_col, y=y_col, title=f"Scatterplot of {x_col} vs {y_col}"))
                        except Exception as e:
                            st.warning(f"Could not generate scatter plot for {x_col} vs {y_col}: {e}")

            # Multivariate Analysis
            st.write("### 🔥 Multivariate Analysis")
            if len(numeric_columns) > 2:
                st.write("#### Pairplot")
                try:
                    fig = sns.pairplot(df[numeric_columns])
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Error generating pairplot: {e}")

                st.write("#### Correlation Heatmap")
                try:
                    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
                    if not numeric_df.empty:
                        fig, ax = plt.subplots()
                        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
                        st.pyplot(fig)
                    else:
                        st.write("No numeric data available for correlation heatmap.")
                except Exception as e:
                    st.write(f"Error generating heatmap: {e}")

            # AI-Generated Insights
            if st.button("✨ Generate AI Insights"):
                insights = generate_insights(df)
                st.subheader("🔍 AI-Generated Insights")
                st.write(insights)

# Run the chatbot mode
chatbot_mode()
