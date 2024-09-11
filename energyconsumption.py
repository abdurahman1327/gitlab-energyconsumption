import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Streamlit settings
st.set_page_config(page_title="Energy and App Classification", page_icon="⚡", layout="wide")

# Sample data
simple_app_data = {
    'time': [
        "02:30:40", "02:30:43", "02:30:46", "02:30:49", "02:30:52", "02:30:55",
        "02:30:58", "02:31:01", "02:31:04", "02:31:07", "02:31:10", "02:31:13",
        "02:31:16", "02:31:19", "02:31:22", "02:31:25", "02:31:28", "02:31:31",
        "02:31:34", "02:31:37", "02:31:40", "02:31:43", "02:31:46", "02:31:49",
        "02:31:52", "02:31:55", "02:31:58", "02:32:01", "02:32:04", "02:32:07",
        "02:32:10", "02:32:13", "02:32:16", "02:32:19", "02:32:22", "02:32:25",
        "02:32:28", "02:32:31", "02:32:34", "02:32:37", "02:32:40"
    ],
    'cpu_usage': [0.17, 0.00, 0.00, 0.00, 0.00, 0.17, 0.00, 0.17, 0.17, 0.00, 0.00, 0.17,
                  0.00, 0.00, 0.00, 0.00, 0.17, 0.17, 0.00, 0.00, 0.33, 0.17, 0.00, 0.00,
                  0.17, 0.00, 0.00, 0.00, 0.17, 0.00, 0.00, 0.17, 0.00, 0.17, 0.00, 0.00,
                  0.00, 0.17, 0.17, 0.00, 0.33],
    'memory_usage': [6141972]*40
}

matrix_app_data = {
    'time': [
        "03:01:22", "03:01:25", "03:01:28", "03:01:31", "03:01:34", "03:01:37",
        "03:01:40", "03:01:43", "03:01:46", "03:01:49", "03:01:52", "03:01:55",
        "03:01:58", "03:02:01", "03:02:04", "03:02:07", "03:02:10", "03:02:13",
        "03:02:16", "03:02:19", "03:02:22", "03:02:25", "03:02:28", "03:02:31",
        "03:02:34", "03:02:37", "03:02:40", "03:02:43", "03:02:46", "03:02:49",
        "03:02:52", "03:02:55", "03:02:58", "03:03:01", "03:03:04", "03:03:07",
        "03:03:10", "03:03:13", "03:03:16", "03:03:19", "03:03:22"
    ],
    'cpu_usage': [0.00, 0.00, 0.00, 0.17, 0.00, 0.17, 0.00, 0.00, 0.17, 0.00, 0.00, 0.00,
                  0.17, 0.00, 0.00, 0.17, 0.00, 0.00, 0.17, 0.00, 0.00, 0.17, 0.17, 0.00,
                  0.17, 0.00, 0.00, 0.17, 0.00, 0.17, 0.17, 0.00, 0.00, 0.17, 0.00, 0.00,
                  0.17, 0.00, 0.17, 0.00, 0.17],
    'memory_usage': [6129092]*40
}

# Convert both datasets to DataFrames
df_simple = pd.DataFrame(simple_app_data)
df_matrix = pd.DataFrame(matrix_app_data)

# Combine the datasets and preprocess
df_simple['app_type'] = 'simple'
df_matrix['app_type'] = 'matrix'
df_combined = pd.concat([df_simple, df_matrix], ignore_index=True)
df_combined['time'] = pd.to_datetime(df_combined['time'], format='%H:%M:%S')
df_combined['time_diff'] = df_combined['time'].diff().dt.total_seconds().fillna(0)

# Assumed power consumption values (in Watts)
cpu_power_watt = 50
memory_power_watt = 5
df_combined['energy_consumption_joules'] = (df_combined['cpu_usage'] * cpu_power_watt + memory_power_watt) * df_combined['time_diff']

# Encoding app type
label_encoder = LabelEncoder()
df_combined['app_type_encoded'] = label_encoder.fit_transform(df_combined['app_type'])

# Features and targets
X = df_combined[['cpu_usage', 'memory_usage', 'energy_consumption_joules']]
y_energy = df_combined['energy_consumption_joules']
y_app_type = df_combined['app_type_encoded']

# Train-test split
X_train, X_test, y_energy_train, y_energy_test = train_test_split(X, y_energy, test_size=0.2, random_state=42)
_, _, y_app_type_train, y_app_type_test = train_test_split(X, y_app_type, test_size=0.2, random_state=42)

# Train models
energy_model = RandomForestRegressor(n_estimators=100, random_state=42)
energy_model.fit(X_train, y_energy_train)

app_type_model = RandomForestClassifier(n_estimators=100, random_state=42)
app_type_model.fit(X_train, y_app_type_train)

# Make predictions
y_energy_pred = energy_model.predict(X_test)
y_app_type_pred = app_type_model.predict(X_test)

# Metrics for energy prediction
mse = mean_squared_error(y_energy_test, y_energy_pred)
r2 = r2_score(y_energy_test, y_energy_pred)

# Metrics for app type classification
accuracy = accuracy_score(y_app_type_test, y_app_type_pred)
classification_report_str = classification_report(y_app_type_test, y_app_type_pred, target_names=label_encoder.classes_)

# Streamlit Layout
st.title("⚡ Energy and App Classification Dashboard")

st.markdown("## Model Metrics")
col1, col2 = st.columns(2)
col1.metric("Energy Prediction - MSE", f"{mse:.2f}")
col1.metric("Energy Prediction - R² Score", f"{r2:.2f}")
col2.metric("App Type Classification - Accuracy", f"{accuracy:.2f}")
st.text("Classification Report:")
st.text(classification_report_str)

st.markdown("## Actual vs Predicted Energy Consumption")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_energy_test.values, label='Actual Energy Consumption', marker='o')
ax.plot(y_energy_pred, label='Predicted Energy Consumption', marker='x')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Energy Consumption (Joules)')
ax.set_title('Actual vs Predicted Energy Consumption')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Adding user input section
st.markdown("## User Input for New Predictions")
cpu_usage_input = st.slider('CPU Usage', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
memory_usage_input = st.number_input('Memory Usage (in MB)', min_value=1000, max_value=10000, value=6141972)
time_diff_input = st.slider('Time Difference (seconds)', min_value=0, max_value=300, value=60)

# Calculating energy consumption based on user input
user_energy_consumption = (cpu_usage_input * cpu_power_watt + memory_power_watt) * time_diff_input

# Making predictions based on user input
user_input_data = np.array([[cpu_usage_input, memory_usage_input, user_energy_consumption]])
predicted_app_type = app_type_model.predict(user_input_data)
predicted_app_type_label = label_encoder.inverse_transform(predicted_app_type)

# Displaying the predicted results
st.markdown("### Prediction Results")
st.write(f"Predicted Energy Consumption (Joules): {user_energy_consumption:.2f}")
st.write(f"Predicted App Type: {predicted_app_type_label[0]}")

