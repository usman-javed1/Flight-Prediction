import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Predefined lists
DESTINATION_AIRPORT = ['MSP', 'DFW', 'ATL', 'IAH', 'ORD', 'PHX', 'DEN', 'LAX', 'SFO', 'LAS']
ORIGIN_AIRPORT = ['SFO', 'LAS', 'DEN', 'LAX', 'PHX', 'ORD', 'DFW', 'MSP', 'IAH', 'ATL']
AIRLINE = ['DL', 'NK', 'AA', 'UA', 'US', 'F9', 'OO', 'WN', 'VX', 'MQ', 'EV', 'B6']

# Load Models
def load_model(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

nb_model = load_model('nb_model_joblib.pkl') 
km_model = load_model('km_model_joblib.pkl')


def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

data = load_data('final_dataset.csv')

# Streamlit App
def main():
    st.title("Flight Arrival Prediction and Clustering")
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to", ["Prediction", "Data Visualization"])

    if section == "Prediction":
        st.header("Predict Arrival Time and Cluster")
        
        # Input Form
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
            # Input DataFrame
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
            df_input = pd.DataFrame([input_data])

            # Display Input Data
            st.subheader("Your Input Data:")
            st.dataframe(df_input)

            # Predictions
            if nb_model is not None:
                prediction = nb_model.predict(df_input)[0]
                st.markdown("### Arrival Time Prediction (Naive Bayes):")
                result_mapping = {0: "High", 1: "Low", 2: "Medium", 3: "Too High"}
                result = result_mapping.get(prediction, "Unknown")
                st.markdown(f"<h3 style='color:blue;'>Arrival Time: {result}</h3>", unsafe_allow_html=True)

            if km_model is not None:
                cluster = km_model.predict(df_input)[0]
                st.markdown("### Clustering Result (K-Means):")
                st.markdown(f"<h3 style='color:green;'>Assigned Cluster: {cluster}</h3>", unsafe_allow_html=True)

            # Visualizations for Input Data in Dataset
            with st.spinner("Generating graphs..."):
                if data is not None:
                    # Cluster Visualization
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
                        ax.scatter(df_input['DISTANCE'], df_input['DEPARTURE_DELAY'], color='red', s=200, label='Your Input')
                        ax.set_title("Clusters with Your Input Data Point")
                        ax.legend()
                        st.pyplot(fig)

                    # Departure Delay vs Arrival Time
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.boxplot(x=data['ARRIVAL_TIME'], y=data['DEPARTURE_DELAY'], data=data, ax=ax, palette='coolwarm')
                    ax.scatter(prediction, departure_delay, color='red', s=200, label='Your Input')
                    ax.set_title("Departure Delay vs Arrival Time")
                    st.pyplot(fig)

    elif section == "Data Visualization":
        st.header("Data Visualization")
        if data is not None:
            with st.spinner("Generating data visualizations..."):
                st.subheader("Dataset Overview")
                st.dataframe(data.head())

                # General Visualizations
                st.subheader("General Visualizations:")
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(data['DISTANCE'], kde=True, color='blue', ax=ax)
                ax.set_title("Distance Distribution")
                st.pyplot(fig)

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x='DAY_OF_WEEK', y='DEPARTURE_DELAY', data=data, palette='coolwarm', ax=ax)
                ax.set_title("Departure Delay by Day of the Week")
                st.pyplot(fig)

                # K-Means Cluster Visualizations
                if km_model is not None:
                    st.subheader("Cluster Visualizations:")
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

if __name__ == "__main__":
    main()
