import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

def compare_to_history(data:pd.DataFrame, sensors:list, references:dict, scaler:object) -> dict:
  """Function to compare input data to recorded data"""

  std_history ={
    k:v for k,v in zip(scaler.feature_names_in_, np.sqrt(scaler.mean_))
  }

  datetime_info = pd.Series(data.index)

  dayName = datetime_info.dt.day_name()[0]
  hour = datetime_info.dt.hour[0]

  msg_sensor = ""
  for col in sensors:
    history_mean = references[col].loc[dayName][hour]
    std_hist = std_history[col]
    current_value = data[col].values[0]

    if not history_mean-std_hist < current_value < history_mean+std_hist:
      diff_value = current_value - history_mean
      msg_sensor += f"- Unusual value of {col} from record history, There is difference of {diff_value:.2f} point."
      msg_sensor += "\n"
  msg_info = "Current condition is Ok, all sensors behave normally" if len(msg_sensor)<1 else msg_sensor

  return msg_info

def make_prediction(data_to_predict: pd.DataFrame, data_:pd.DataFrame, ts):
  """Function to make prediction and display it to the user"""

  prediction = model.predict(scaler.transform(data_to_predict))
  prediction_str = "This condition is Normal" if prediction == 1 else "This condition is not Normal"
  reason = compare_to_history(data_to_predict, feature_names, references, scaler)

  # Display prediction only when button is clicked
  if st.button("Predict"):
    if prediction == 1:
      st.dataframe(data_to_predict)
      st.info(prediction_str)
      
      for col in data.columns[:5]:
        fig = plt.figure(figsize=(15, 3.5))
        plt.style.use("fivethirtyeight")
        plt.plot(data_.index, data_[col], label=col, linewidth=1)
        plt.xticks(rotation=30)
        plt.axvline(x=ts, c="r", label="choosen timestamp", linewidth=1)
        plt.title(col)
        plt.legend()
        st.pyplot(fig)
    
    else:
      st.dataframe(data_to_predict)
      st.error(prediction_str)
      st.error(reason)
      for col in data.columns[:5]:
        fig = plt.figure(figsize=(10, 3.5))
        plt.style.use("fivethirtyeight")
        plt.plot(data_.index, data_[col], label=col, linewidth=1)
        plt.xticks(rotation=30)
        plt.axvline(x=ts, c="r", label="choosen timestamp", linewidth=1)
        plt.title(col)
        plt.legend()
        st.pyplot(fig)

# header
st.header("Welcome to Sensor Monitoring Dashboard!😎")

# load data
device = st.selectbox("Choose data", ("device_b", "device_c", "device_a"))
data = pd.read_parquet(f"data/{device}.parquet")

# plot data
selected_cols = st.multiselect("Select columns for plotting:", data.columns[:5])
if selected_cols:
  for col in selected_cols:
    st.line_chart(data[[col]])

# load model & utilities
with open("models/iso_forest.bin", "rb") as f:
  model = pickle.load(f)
with open ("data/sensor_features.bin", "rb") as f:
  feature_names = pickle.load(f)
with open ("data/references.bin", "rb") as f:
  references = pickle.load(f)
with open ("data/scaler.bin", "rb") as f:
  scaler = pickle.load(f)

# choose timestamp
data_timestamp = st.selectbox(
                    "Choose timestamp",
                    data.index
                  )
data_to_predict = data[feature_names].loc[data_timestamp].to_frame().T

# make prediction
make_prediction(data_to_predict, data, data_timestamp)