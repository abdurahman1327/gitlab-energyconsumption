import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import timedelta



# Setting the Streamlit theme and overall color palette for visual consistency
st.set_page_config(page_title="Energy Consumption Dashboard", page_icon="⚡", layout="wide")
sns.set_palette("RdYlGn")  # Red-Yellow-Green color palette for energy usage

# Sample data for energy consumption
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

# Sidebar for time range selection
st.sidebar.title("⚙️ Filters")
start_time = st.sidebar.time_input('Start Time', df['time'].min().time())
end_time = st.sidebar.time_input('End Time', (df['time'].min() + timedelta(minutes=1)).time())  # Halve the time range

# Filter the DataFrame based on the selected time range
df_filtered = df[(df['time'].dt.time >= start_time) & (df['time'].dt.time <= end_time)]

# Dashboard Title
st.title('⚡ Pipeline Energy Consumption Dashboard')

# KPI Section: Show key metrics
st.markdown("## Key Metrics")
col1, col2, col3 = st.columns(3)

total_energy = df_filtered['energy_consumption'].sum()
avg_cpu = df_filtered['cpu_usage'].mean()
avg_memory = df_filtered['memory_usage'].mean()

col1.metric("⚡ Total Energy Consumption", f"{total_energy:.2f} kWh", delta=f"{(total_energy - df['energy_consumption'].mean()):.2f}")
col2.metric("💻 Avg CPU Usage", f"{avg_cpu:.2f}%", delta=f"{avg_cpu - df['cpu_usage'].mean():.2f}")
col3.metric("📊 Avg Memory Usage", f"{avg_memory / 1e6:.2f} MB", delta=f"{avg_memory - df['memory_usage'].mean():.2f}")

# Mini-Plots Section
st.markdown("### Metrics Over Time")

fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True)

# Plot Energy Consumption
axs[0, 0].plot(df_filtered['time'], df_filtered['energy_consumption'], color='tab:blue')
axs[0, 0].set_title('Energy Consumption (kWh)')
axs[0, 0].set_ylabel('Energy Consumption (kWh)')

# Plot CPU Usage
axs[0, 1].plot(df_filtered['time'], df_filtered['cpu_usage'], color='tab:orange')
axs[0, 1].set_title('CPU Usage (%)')
axs[0, 1].set_ylabel('CPU Usage (%)')

# Plot Memory Usage
axs[1, 0].plot(df_filtered['time'], df_filtered['memory_usage'] / 1e6, color='tab:green')
axs[1, 0].set_title('Memory Usage (MB)')
axs[1, 0].set_ylabel('Memory Usage (MB)')

# Plot Cumulative Values
df_filtered['cumulative_energy'] = df_filtered['energy_consumption'].cumsum() / 2  # Halve the values
df_filtered['cumulative_cpu'] = df_filtered['cpu_usage'].cumsum() / 2  # Halve the values
df_filtered['cumulative_memory'] = (df_filtered['memory_usage'] / 1e6).cumsum() / 2  # Halve the values

axs[1, 1].plot(df_filtered['time'], df_filtered['cumulative_energy'], color='tab:blue', label='Cumulative Energy Consumption')
axs[1, 1].plot(df_filtered['time'], df_filtered['cumulative_cpu'], color='tab:orange', linestyle='--', label='Cumulative CPU Usage')
axs[1, 1].plot(df_filtered['time'], df_filtered['cumulative_memory'], color='tab:green', linestyle=':', label='Cumulative Memory Usage')
axs[1, 1].set_title('Cumulative Metrics')
axs[1, 1].set_ylabel('Cumulative Value')
axs[1, 1].legend()

# Improve layout
plt.tight_layout()
st.pyplot(fig)

# Clearer Efficiency Overview
st.markdown("### Efficiency Overview")

# Define benchmarks or targets
target_cpu_usage = 0.20
target_memory_usage = 6142000

# Compute efficiency
cpu_efficiency = avg_cpu / target_cpu_usage * 100
memory_efficiency = avg_memory / target_memory_usage * 100

col4, col5 = st.columns(2)

col4.metric("CPU Efficiency", f"{cpu_efficiency:.1f}%", delta=f"Target: {target_cpu_usage * 100:.1f}%")
col5.metric("Memory Efficiency", f"{memory_efficiency:.1f}%", delta=f"Target: {target_memory_usage / 1e6:.1f} MB")

# Add progress bars
st.markdown("#### CPU Usage Progress")
st.progress(int(cpu_efficiency))

st.markdown("#### Memory Usage Progress")
st.progress(int(memory_efficiency))

# Energy-saving tips based on consumption
st.markdown("## Energy-Saving Insights 💡")
if total_energy > 0.5:
    st.success("Consider optimizing pipelines by reducing memory or CPU usage to save energy.")
else:
    st.info("Pipelines are running efficiently with low energy consumption.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """<div style="text-align: center; font-size: 12px;">
    Gitlab API
    </div>""",
    unsafe_allow_html=True
)

