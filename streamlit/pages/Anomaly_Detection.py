import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from snowflake.snowpark.context import get_active_session

from snowflake.ml.registry import Registry

st.set_page_config(layout="wide")

from datetime import datetime, timedelta

# Get the current credentials
session = get_active_session()
session.sql("USE ROLE SYSADMIN")

st.title("Run inference using a trained Autoencoder model to detect anomalies in sensor data")

# Load the data from Snowflake
queried_data = session.sql("SELECT * FROM AE_TEST").to_pandas()
queried_data['MEASURE_TS'] = pd.to_datetime(queried_data['MEASURE_TS'], format='%m/%d/%y %H:%M')

# Get yesterday's date
yesterday = datetime.now() - timedelta(days=1)
yesterday_date = yesterday.date()

# Update the MEASURE_TS column to have yesterday's date but keep the original time
queried_data['MEASURE_TS'] = queried_data['MEASURE_TS'].apply(
    lambda dt: dt.replace(year=yesterday_date.year, month=yesterday_date.month, day=yesterday_date.day)
)

# If you need the MEASURE_TS column back in the original string format
queried_data['MEASURE_TS'] = queried_data['MEASURE_TS'].dt.strftime('%m/%d/%y %H:%M')

# Show the data table in Streamlit
st.subheader("Sensor Data")
#st.write(queried_data)

# Button to trigger inference
if st.button("Click button to carry inference and detect Anomalies"):
    with st.spinner("Getting predictions..."):
        # Create a Snowpark DataFrame
        X_train_snowdf = session.create_dataframe(queried_data)

        # Set up the model
        db = session.get_current_database()
        schema = session.get_current_schema()
        model_name = "AnomalyDetection_Model_1"
        model_version = "v1"
        reg = Registry(session=session) 
        modelversion = reg.get_model(model_name).version(model_version)

        # Run inference
        X_pred = modelversion.run(X_train_snowdf, function_name="predict")

        # Rename the columns for better readability
        rename_mapping = {
            "feature_0": "TEMPERATURE_LOSS_MAE",
            "feature_1": "VIBRATION_LOSS_MAE",
            "feature_2": "MOTOR_RPM_LOSS_MAE",
            "feature_3": "MOTOR_AMPS_LOSS_MAE"
        }
        X_predpdtest = X_pred.to_pandas()
        X_predpdtest = X_predpdtest.rename(columns=rename_mapping)

        # Display the results
        st.subheader("Inference Results")
        st.write(X_predpdtest)

        queried_traindata = session.sql("SELECT * FROM AE_TRAIN")
        X_predTrain = modelversion.run(queried_traindata, function_name="predict")
        
        X_predpd = X_predTrain.to_pandas()
        X_predpd = X_predpd.rename(columns=rename_mapping)


        # Get reconstruction loss threshold for each sensor from the training data
        
        Threshold_TEMPERATURE= np.max(X_predpd['TEMPERATURE_LOSS_MAE'])
        print("Reconstruction error threshold: ", Threshold_TEMPERATURE)
        Threshold_VIBRATION= np.max(X_predpd['VIBRATION_LOSS_MAE'])
        print("Reconstruction error threshold: ", Threshold_VIBRATION)
        Threshold_MOTOR_AMPS= np.max(X_predpd['MOTOR_AMPS_LOSS_MAE'])
        print("Reconstruction error threshold: ", Threshold_MOTOR_AMPS)
        Threshold_MOTOR_RPM= np.max(X_predpd['MOTOR_RPM_LOSS_MAE'])
        print("Reconstruction error threshold: ", Threshold_MOTOR_RPM)

        X_predpdtest['Threshold_VIBRATION']= Threshold_VIBRATION
        X_predpdtest['Threshold_MOTOR_AMPS']= Threshold_MOTOR_AMPS
        X_predpdtest['Threshold_MOTOR_RPM']= Threshold_MOTOR_RPM
        X_predpdtest['Threshold_TEMPERATURE']= Threshold_TEMPERATURE

        #Determine anomaly in each sensor
        X_predpdtest['Anomaly_in_vibration_sensor'] = X_predpdtest['VIBRATION_LOSS_MAE'] > X_predpdtest['Threshold_VIBRATION']
        X_predpdtest['Anomaly_in_temperature_sensor'] = X_predpdtest['TEMPERATURE_LOSS_MAE'] > X_predpdtest['Threshold_TEMPERATURE']
        X_predpdtest['Anomaly_in_motor_amps_sensor'] = X_predpdtest['MOTOR_AMPS_LOSS_MAE'] > X_predpdtest['Threshold_MOTOR_AMPS']
        X_predpdtest['Anomaly_in_motor_rpm_sensor'] = X_predpdtest['MOTOR_RPM_LOSS_MAE'] > X_predpdtest['Threshold_MOTOR_RPM']
        
        #Combine anomalies across all sensors
        snow_pred_test_final = X_predpdtest[['Anomaly_in_vibration_sensor', 'Anomaly_in_temperature_sensor', 'Anomaly_in_motor_amps_sensor','Anomaly_in_motor_rpm_sensor']]
        
        #Set the Threshold calculated for the training data 
        X_predpd['Threshold_VIBRATION']= Threshold_VIBRATION
        X_predpd['Threshold_MOTOR_AMPS']= Threshold_MOTOR_AMPS
        X_predpd['Threshold_MOTOR_RPM']= Threshold_MOTOR_RPM
        X_predpd['Threshold_TEMPERATURE']= Threshold_TEMPERATURE
        
        
            
        #Determine anomaly
        X_predpd['Anomaly_in_vibration_sensor'] = X_predpd['VIBRATION_LOSS_MAE'] > X_predpd['Threshold_VIBRATION']
        X_predpd['Anomaly_in_temperature_sensor'] = X_predpd['TEMPERATURE_LOSS_MAE'] > X_predpd['Threshold_TEMPERATURE']
        X_predpd['Anomaly_in_motor_amps_sensor'] = X_predpd['MOTOR_AMPS_LOSS_MAE'] > X_predpd['Threshold_MOTOR_AMPS']
        X_predpd['Anomaly_in_motor_rpm_sensor'] = X_predpd['MOTOR_RPM_LOSS_MAE'] > X_predpd['Threshold_MOTOR_RPM']
        
        scored = pd.concat([X_predpd, X_predpdtest])
        anomaly_sensors = ['Anomaly_in_vibration_sensor', 'Anomaly_in_temperature_sensor','Anomaly_in_motor_amps_sensor','Anomaly_in_motor_rpm_sensor']
        scored['Anomaly']=scored[anomaly_sensors].any(axis=1)
        #Concat train and test data and plot the anomalies detected
        merge_test = session.table("AE_TEST").to_pandas()   

        merge_test.set_index('MEASURE_TS', drop=True, inplace=True)
        merge_train = session.table("AE_TRAIN").to_pandas()
        merge_train.set_index('MEASURE_TS', drop=True, inplace=True)
    
        merge_concat=pd.concat([merge_train, merge_test])
        
        merge_concat.reindex()
        scored.index = merge_concat.index
        
        merge_concat['Anomaly']=scored['Anomaly']
        
        anomalies=merge_concat[merge_concat['Anomaly']==True].index
        
        st.write(merge_concat)

        
        merge_concat.index = pd.to_datetime(merge_concat.index)

        # Plot the data
        fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
        
        # Plot each sensor's data
        ax.plot(merge_concat.index, merge_concat['VIBRATION'], label='Vibration', color='blue', linewidth=1)
        ax.plot(merge_concat.index, merge_concat['MOTOR_RPM'], label='Motor RPM', color='gray', linewidth=1)
        ax.plot(merge_concat.index, merge_concat['MOTOR_AMPS'], label='Motor Amps', color='green', linewidth=1)
        ax.plot(merge_concat.index, merge_concat['TEMPERATURE'], label='Temperature', color='black', linewidth=1)
        
        # Plot anomalies
        anomaly_indices = merge_concat[merge_concat['Anomaly'] == True].index
        for anomaly in anomaly_indices:
            ax.axvline(x=anomaly, color='red', alpha=0.3)
        
        # Customize the plot
        plt.legend(loc='lower left')
        ax.set_title('Sensor Anomaly Detection', fontsize=16)
        ax.set_xlabel('Timestamp', fontsize=12)
        ax.set_ylabel('Sensor Values', fontsize=12)
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M:%S'))  # Format datetime
        
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid(alpha=0.3)
        plt.tight_layout()

        st.pyplot(fig)
    
