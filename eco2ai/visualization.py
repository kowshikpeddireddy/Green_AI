import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
data = pd.read_csv("emission.csv")

# Visualization for Power Consumption
plt.figure(figsize=(10, 6))
sns.histplot(data['power_consumption(kWh)'], bins=20, kde=True)
plt.title('Distribution of Power Consumption')
plt.xlabel('Power Consumption (kWh)')
plt.ylabel('Frequency')
plt.show()

# Visualization for CO2 Emissions
plt.figure(figsize=(10, 6))
sns.histplot(data['CO2_emissions(kg)'], bins=20, kde=True)
plt.title('Distribution of CO2 Emissions')
plt.xlabel('CO2 Emissions (kg)')
plt.ylabel('Frequency')
plt.show()
# Visualization for CPU
plt.figure(figsize=(12, 6))
sns.countplot(data=data, x='CPU_name')
plt.title('CPU Distribution')
plt.xlabel('CPU Model')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Visualization for GPU
plt.figure(figsize=(10, 6))
sns.countplot(data['GPU_name'])
plt.title('GPU Distribution')
plt.xlabel('GPU Name')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Visualization for OS
plt.figure(figsize=(10, 6))
sns.countplot(data['OS'])
plt.title('Operating System Distribution')
plt.xlabel('Operating System')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Visualization for Region/Country
plt.figure(figsize=(10, 6))
sns.countplot(data['region/country'])
plt.title('Region/Country Distribution')
plt.xlabel('Region/Country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
