import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import timedelta

# Customizing Seaborn style for better aesthetics
sns.set_theme(style="whitegrid")

# Sample data extracted from the given statistics
data = {
    'time': [
        "02:30:40", "02:30:43", "02:30:46", "02:30:49", "02:30:52", "02:30:55",
        "02:30:58", "02:31:01", "02:31:04", "02:31:07", "02:31:10", "02:31:13",
        "02:31:16", "02:31:19", "02:31:22", "02:31:25", "02:31:28", "02:31:31",
        "02:31:34", "02:31:37", "02:31:40", "02:31:43", "02:31:46", "02:31:49",
        "02:31:52", "02:31:55", "02:31:58", "02:32:01", "02:32:04", "02:32:07",
        "02:32:10", "02:32:13", "02:32:16", "02:32:19", "02:32:22", "02:32:25",
        "02:32:28", "02:32:31", "02:32:34", "02:32:37", "02:32:40"
    ],
    'cpu_usage': [
        0.17, 0.00, 0.00, 0.00, 0.00, 0.17, 0.00, 0.17, 0.17, 0.00, 0.00, 0.17,
        0.00, 0.00, 0.00, 0.00, 0.17, 0.17, 0.00, 0.00, 0.33, 0.17, 0.00, 0.00,
        0.17, 0.00, 0.00, 0.00, 0.17, 0.00, 0.00, 0.17, 0.00, 0.17, 0.00, 0.00,
        0.00, 0.17, 0.17, 0.00, 0.33
    ],
    'memory_usage': [
        6141972, 6141972, 6141972, 6141972, 6141972, 6141972, 6141972, 6141972,
        6141972, 6141972, 6141972, 6141972, 6141972, 6141972, 6141972, 6141972,
        6141972, 6141972, 6141972, 6141972, 6141972, 6141972, 6141972, 6141972,
        6141972, 6141972, 6141972, 6141972, 6141972, 6141972, 6141972, 6141972,
        6141972, 6141972, 6141972, 6141972, 6141972, 6141972, 6141972, 6141972,
        6141972
    ]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)
df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S')

# Calculate energy consumption (a simplified formula for demonstration)
df['energy_consumption'] = df['cpu_usage'] * df['memory_usage'] / 1e6

# Adding a time range slider for filtering data
start_time = st.sidebar.time_input('Start Time', df['time'].min().time())
end_time = st.sidebar.time_input('End Time', (df['time'].min() + timedelta(minutes=2)).time())

# Filter the DataFrame based on the selected time range
df_filtered = df[(df['time'].dt.time >= start_time) & (df['time'].dt.time <= end_time)]

# Create Streamlit dashboard
st.title('🚀 Innovative Energy Consumption Dashboard for Pipelines')

# Display KPIs at the top
st.markdown("### Key Metrics")
col1, col2, col3 = st.columns(3)

total_energy = df_filtered['energy_consumption'].sum()
avg_cpu = df_filtered['cpu_usage'].mean()
avg_memory = df_filtered['memory_usage'].mean()

col1.metric("⚡ Total Energy Consumption", f"{total_energy:.2f} units")
col2.metric("💻 Avg CPU Usage", f"{avg_cpu:.2f}%")
col3.metric("📊 Avg Memory Usage", f"{avg_memory / 1e6:.2f} MB")

# Create a line chart to visualize energy consumption over time
st.markdown("### Energy Consumption Over Time")
st.line_chart(df_filtered.set_index('time')[['energy_consumption', 'cpu_usage', 'memory_usage']])

# Add custom Matplotlib plot with insights
st.markdown("### Detailed Visualization")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_filtered['time'], df_filtered['energy_consumption'], label='Energy Consumption', color='green', linewidth=2)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Energy Consumption (arbitrary units)', fontsize=12)
ax.axhline(df_filtered['energy_consumption'].mean(), color='red', linestyle='--', label='Avg Energy Consumption')
ax.fill_between(df_filtered['time'], df_filtered['energy_consumption'], color='lightgreen', alpha=0.4)

# Customize the plot further
ax.legend(loc='upper right')
ax.grid(True)
st.pyplot(fig)

# Insights or energy-saving tips based on data
st.markdown("### Energy-Saving Insights 💡")
if total_energy > 0.5:
    st.success("Consider optimizing your pipelines by reducing memory or CPU usage to save energy.")
else:
    st.info("Your pipelines are already efficient with low energy consumption.")
