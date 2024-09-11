import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import timedelta



# Set Streamlit theme and page configuration
st.set_page_config(page_title="Energy Consumption Dashboard", page_icon="⚡", layout="wide")
st.markdown("""
    <style>
    .reportview-container {
        background-color: #2E2E2E;
    }
    .sidebar .sidebar-content {
        background-color: #1E1E1E;
    }
    .block-container {
        padding: 2rem;
    }
    .css-1v0mbdj {
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

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
end_time = st.sidebar.time_input('End Time', (df['time'].min() + timedelta(minutes=2)).time())

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

# Interactive Line Chart for Energy Consumption
st.markdown("### Energy Consumption Over Time")
st.line_chart(df_filtered.set_index('time')[['energy_consumption']])

# Adding Heatmap for CPU and Memory Utilization
st.markdown("### CPU and Memory Usage Heatmap")

# Create a heatmap with Seaborn
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap([df_filtered['cpu_usage'], df_filtered['memory_usage']], cmap="RdYlGn_r", ax=ax, cbar=True, cbar_kws={'label': 'Usage'}, linewidths=0.5, linecolor='black')
ax.set_yticklabels([])
ax.set_xticklabels(df_filtered['time'].dt.strftime('%H:%M:%S'), rotation=45)

# Hide gridlines
ax.grid(False)

# Use dark background color
ax.set_facecolor('#2E2E2E')
fig.patch.set_facecolor('#2E2E2E')

st.pyplot(fig)

# Enhanced Efficiency Overview
st.markdown("### Efficiency Overview")
col4, col5 = st.columns(2)

# CPU Usage Progress
col4.markdown("#### CPU Usage")
cpu_efficiency = int(avg_cpu)
col4.progress(cpu_efficiency, text=f"CPU Usage: {cpu_efficiency}%")

# Memory Usage Progress
col5.markdown("#### Memory Usage")
memory_efficiency = avg_memory / max(df['memory_usage']) * 100
col5.progress(int(memory_efficiency), text=f"Memory Usage: {int(memory_efficiency)}%")

# Energy-saving tips based on consumption
st.markdown("## Energy-Saving Insights 💡")
if total_energy > 0.5:
    st.success("Consider optimizing pipelines by reducing memory or CPU usage to save energy.")
else:
    st.info("Pipelines are running efficiently with low energy consumption.")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """<div style="text-align: center; font-size: 12px; color: white;">
    Made with 💻 by [Your Name] for the Hackathon Project.
    </div>""",
    unsafe_allow_html=True
)
