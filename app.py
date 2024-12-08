import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from helping_module import *
import numpy as np

# Predefined lists
DESTINATION_AIRPORT = ['MSP', 'DFW', 'ATL', 'IAH', 'ORD', 'PHX', 'DEN', 'LAX', 'SFO', 'LAS']
ORIGIN_AIRPORT = ['SFO', 'LAS', 'DEN', 'LAX', 'PHX', 'ORD', 'DFW', 'MSP', 'IAH', 'ATL']
AIRLINE = ['DL', 'NK', 'AA', 'UA', 'US', 'F9', 'OO', 'WN', 'VX', 'MQ', 'EV', 'B6']

# Load Pretrained Models
def load_model(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

nb_model = load_model('nb_model_joblib.pkl')
km_model = load_model('km_model_joblib.pkl')

# Load Dataset
@st.cache
def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data('final_dataset.csv')

# Streamlit App
def main():
    st.title("Flight Arrival Time Prediction and Clustering")
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", ["Prediction", "Data Visualization"])

    if section == "Prediction":
        # Input Columns
        st.header("Predict Arrival Time and Cluster")
        col1, col2 = st.columns(2)

        with col1:
            year = st.number_input("Year", min_value=2000, max_value=2024, value=2023)
            month = st.number_input("Month", min_value=1, max_value=12, value=1)
            day = st.number_input("Day", min_value=1, max_value=31, value=1)
            day_of_week = st.number_input("Day of Week", min_value=1, max_value=7, value=1)
            airline = st.selectbox("Airline", AIRLINE)
            origin_airport = st.selectbox("Origin Airport", ORIGIN_AIRPORT)

        with col2:
            destination_airport = st.selectbox("Destination Airport", DESTINATION_AIRPORT)
            scheduled_departure = st.number_input("Scheduled Departure (float)", value=0.0, format="%.2f")
            departure_time = st.number_input("Departure Time (float)", value=0.0, format="%.2f")
            departure_delay = st.number_input("Departure Delay (float)", value=0.0, format="%.2f")
            taxi_out = st.number_input("Taxi Out (float)", value=0.0, format="%.2f")
            wheels_off = st.number_input("Wheels Off (float)", value=0.0, format="%.2f")

        # Additional Inputs
        col3, col4 = st.columns(2)

        with col3:
            scheduled_time = st.number_input("Scheduled Time (float)", value=0.0, format="%.2f")
            elapsed_time = st.number_input("Elapsed Time (float)", value=0.0, format="%.2f")
            air_time = st.number_input("Air Time (float)", value=0.0, format="%.2f")
            distance = st.number_input("Distance (float)", value=0.0, format="%.2f")
            wheels_on = st.number_input("Wheels On (float)", value=0.0, format="%.2f")

        with col4:
            taxi_in = st.number_input("Taxi In (float)", value=0.0, format="%.2f")

        # Submit Button
        if st.button("Predict and Cluster"):
            # Prepare Data for Prediction
            input_data = {
                'YEAR': year, 'MONTH': month, 'DAY': day, 
                'DAY_OF_WEEK': day_of_week, 'AIRLINE': airline,
                'ORIGIN_AIRPORT': origin_airport, 
                'DESTINATION_AIRPORT': destination_airport,
                'SCHEDULED_DEPARTURE': scheduled_departure,
                'DEPARTURE_TIME': departure_time,
                'DEPARTURE_DELAY': departure_delay,
                'TAXI_OUT': taxi_out, 'WHEELS_OFF': wheels_off,
                'SCHEDULED_TIME': scheduled_time,
                'ELAPSED_TIME': elapsed_time,
                'AIR_TIME': air_time, 'DISTANCE': distance,
                'WHEELS_ON': wheels_on, 'TAXI_IN': taxi_in
            }

            df = pd.DataFrame([input_data])
            st.write("Input Data:")
            st.dataframe(df)

            if nb_model is None or km_model is None:
                st.error("One or more models could not be loaded. Please check the model files.")
                return

            # Make Prediction (Naive Bayes)
            prediction = nb_model.predict(df)[0]

            # Make Cluster Prediction (K-Means)
            cluster = km_model.predict(df)[0]

            # Display Prediction and Cluster
            st.markdown("### Prediction Result:")
            result_mapping = {
                0: "High Arrival time",
                1: "Low Arrival time",
                2: "Medium Arrival time",
                3: "Too High Arrival time",
            }
            result = result_mapping.get(prediction, "Unknown")
            st.markdown(f"<h1 style='color:blue;'>Arrival Time Prediction: {result}</h1>", unsafe_allow_html=True)
            st.markdown(f"<h2>Cluster Assigned: {cluster}</h2>")

            # Visualize where the data point lies in clusters
            st.markdown("### Data Point and Clustering Visualization:")
            with st.spinner("Preparing cluster visualizations..."):
                if data is not None:
                    # Add cluster labels to the dataset
                    cluster_data = data.copy()
                    cluster_data['Cluster'] = km_model.predict(data)

                    # Visualize clusters
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(
                        x=cluster_data['DISTANCE'],
                        y=cluster_data['DEPARTURE_DELAY'],
                        hue=cluster_data['Cluster'],
                        palette='viridis',
                        alpha=0.7,
                        ax=ax
                    )
                    ax.scatter(df['DISTANCE'], df['DEPARTURE_DELAY'], color='red', s=200, label='Your Data Point')
                    ax.set_title("Clusters by Distance and Departure Delay")
                    ax.legend()
                    st.pyplot(fig)

    elif section == "Data Visualization":
        st.header("Data Visualization")
        if data is not None:
            with st.spinner("Preparing data visualizations..."):
                st.subheader("Dataset Overview")
                st.dataframe(data.head())

                st.subheader("Cluster Visualization")
                if km_model is not None:
                    cluster_data = data.copy()
                    cluster_data['Cluster'] = km_model.predict(data)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(
                        x=cluster_data['DISTANCE'],
                        y=cluster_data['DEPARTURE_DELAY'],
                        hue=cluster_data['Cluster'],
                        palette='viridis',
                        alpha=0.7,
                        ax=ax
                    )
                    ax.set_title("Clusters by Distance and Departure Delay")
                    st.pyplot(fig)


if __name__ ==  "__main__":
    main()