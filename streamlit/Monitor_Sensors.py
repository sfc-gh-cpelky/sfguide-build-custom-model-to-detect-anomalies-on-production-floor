import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from snowflake.snowpark.context import get_active_session

st.title('Monitor Sensor Data')

# Get the current credentials
session = get_active_session()

session.sql("USE ROLE SYSADMIN")

# Load data from Snowflake table
def load_sensor_data(session, table_name="SENSOR_PREPARED"):
    return session.table(table_name).to_pandas()

# Main Streamlit app
def main():
    st.title("Sensor Data Visualization")
    
    # Load data
    st.write("Loading Sensor Data from Snowflake")
    try:
        sensor_data = load_sensor_data(session)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return

    # Display current data
    st.subheader("Current Sensor Data")
    st.dataframe(sensor_data, use_container_width=True)

    # Ensure the index is in datetime format
    sensor_data['MEASURE_TS'] = pd.to_datetime(sensor_data['MEASURE_TS'])
    sensor_data.set_index('MEASURE_TS', inplace=True)

    # Line plot for all sensor readings over time
    st.subheader("Sensor Readings Over Time")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=sensor_data, ax=ax)
    ax.set_title("Line Plot of Sensor Readings", fontsize=16)
    ax.set_xlabel("Timestamp", fontsize=12)
    ax.set_ylabel("Sensor Values", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Area plot for all sensor readings
    st.subheader("Sensor Readings - Area Plot")
    temp = sensor_data.copy()
    temp.drop(labels=['MEASURE_TS'], axis=1, inplace=True, errors='ignore')  # MEASURE_TS is now the index
    fig, ax = plt.subplots(figsize=(12, 6))
    temp.plot.area(ax=ax)
    ax.set_title("Area Plot of Sensor Readings", fontsize=16)
    ax.set_xlabel("Timestamp", fontsize=12)
    ax.set_ylabel("Sensor Values", fontsize=12)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    

    # Sensor Reading Correlation Heatmap
    st.subheader("Sensor Reading Correlation Heatmap")
    try:
        numeric_data = sensor_data.select_dtypes(include='number')  # Filter numeric columns
        if not numeric_data.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            correlation_matrix = numeric_data.corr()
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
            ax.set_title("Correlation Heatmap of Sensor Readings", fontsize=16)
            st.pyplot(fig)
        else:
            st.warning("No numeric data available for correlation heatmap.")
    except Exception as e:
        st.error(f"Error generating heatmap: {e}")

    

if __name__ == "__main__":
    main()
