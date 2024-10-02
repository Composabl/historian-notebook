# Exploring Composabl Historian Data with Jupyter Notebook

Welcome to this guide on how to explore Composabl Historian data using a Jupyter Notebook. This repository provides a step-by-step tutorial to analyze and visualize Agent training data from Composabl's Historian.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
  - [1. Download Historian Data](#1-download-historian-data)
  - [2. Set Up Your Environment](#2-set-up-your-environment)
  - [3. Launch the Jupyter Notebook](#3-launch-the-jupyter-notebook)
- [Understanding the Data](#understanding-the-data)
- [Data Exploration and Visualization](#data-exploration-and-visualization)
  - [1. Loading the Data](#1-loading-the-data)
  - [2. Data Processing](#2-data-processing)
  - [3. Visualizing Agent Training Data](#3-visualizing-agent-training-data)
- [Additional Resources](#additional-resources)
- [License](#license)

---

## Prerequisites

Before you begin, ensure you have the following:

- **Python 3.7 or later** installed on your machine.
- **Jupyter Notebook** or **JupyterLab** installed. If not, you can install it using `pip`:

  ```bash
  pip install notebook
  ```

- Basic knowledge of Python and data analysis libraries such as `pandas` and `matplotlib`.

## Getting Started

### 1. Download Historian Data

You will need two files from the Composabl UI:

1. **Parquet File**: Contains the main dataset.
2. **`_delta_log` Directory**: Contains transaction logs for the dataset.

**Steps to download:**

- Log in to your Composabl UI.
- Navigate to a training session.
- Click on the Artifacts dropdown and download the Historian files.

![Screenshot by Dropbox Capture](https://github.com/user-attachments/assets/7e190ca5-f149-442b-a47a-f9c22cb3b580)


### 2. Set Up Your Environment

Clone this repository or download the notebook to your local machine.

```bash
git clone https://github.com/composabl/historian-notebook.git
cd historian-notebook
```

Ensure you have the required Python packages installed:

```bash
pip install pandas matplotlib pyarrow
```

### 3. Launch the Jupyter Notebook

Start the Jupyter Notebook server:

```bash
jupyter notebook
```

Open the notebook `composabl historian.ipynb` in your browser.

## Understanding the Data

The Composabl Historian data provides comprehensive logs of Agent training processes, including:

- **Agent Observations**: Data about the agent's environment and state.
- **Training Metrics**: Performance metrics over training iterations.
- **Event Logs**: Detailed logs of events during training sessions.

The data is stored in Parquet format, a columnar storage file format optimized for use with big data processing frameworks.

## Data Exploration and Visualization

### 1. Loading the Data

In the notebook, we'll start by importing the necessary libraries and loading the Parquet file into a pandas DataFrame.

```python
import pandas as pd
import ast 
import matplotlib.pyplot as plt
import json
import numpy as np

# Replace with your actual file paths
df = pd.read_parquet('path/to/your_data.parquet')
# Sort by timestamp
df = df.sort_values(by=['timestamp'])
```

### 2. Data Processing

We'll perform data processing steps to prepare the data for analysis:

#### **a. Filtering Relevant Data**

We focus on specific `category_sub` values that are relevant to Agent training.

```python
# Filter DataFrame for specific 'category_sub' entries
df_data = df[df['category_sub'].isin(['step', 'skill-training', 'skill-training-cycle'])]
```

#### **b. Filtering for 'composabl_obs' in 'data' Column**

We further filter the DataFrame to include only rows where the 'data' column contains 'composabl_obs'.

```python
# Filter df_data where 'data' contains 'composabl_obs' or specific 'category_sub' values
df_data = df_data[
    (df_data['data'].str.contains('composabl_obs', na=False)) |
    (df_data['category_sub'].str.contains('skill-training', na=False)) |
    (df_data['category_sub'].str.contains('skill-training-cycle', na=False))
]
```

#### **c. Converting JSON Strings to Dictionaries**

The 'data' column contains JSON strings that we need to convert to dictionaries.

```python
# Function to convert JSON strings to dictionaries
def convert_to_dict(x):
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return None

df_data['data'] = df_data['data'].apply(convert_to_dict)
```

#### **d. Extracting Fields from 'data'**

We extract relevant fields such as 'skill_name', 'reward', 'obs', and 'cycle' from the 'data' dictionaries.

```python
# Extract 'skill_name' where 'is_done' is in the data
df_data['skill_name'] = df_data['data'].apply(lambda x: x['name'] if isinstance(x, dict) and 'is_done' in x else None)
df_data['skill_name'] = df_data['skill_name'].fillna(method='bfill')

# Extract 'reward' where 'composabl_obs' is in the data
df_data['reward'] = df_data['data'].apply(lambda x: x['teacher_reward'] if isinstance(x, dict) and 'composabl_obs' in x else None)

# Extract 'obs' where 'composabl_obs' is in the data
df_data['obs'] = df_data['data'].apply(lambda x: x['composabl_obs'] if isinstance(x, dict) and 'composabl_obs' in x else None)

# Extract 'cycle' and fill missing values
df_data['cycle'] = df_data['data'].apply(lambda x: x['cycle'] if isinstance(x, dict) and 'cycle' in x else None)
df_data['cycle'] = df_data['cycle'].fillna(method='bfill')

# Filter df_data to include only 'step' entries
df_data = df_data[df_data['category_sub'] == 'step']
```

#### **e. Handling Timestamps**

Ensure that the 'timestamp' column is converted to datetime format and set as the index.

```python
# Convert 'timestamp' to datetime and set as index
df_data['timestamp'] = pd.to_datetime(df_data['timestamp'], unit='ns', errors='coerce')
df_data.set_index('timestamp', inplace=True)
```

### 3. Visualizing Agent Training Data

#### **a. Grouping Data by Runs**

We group the data by 'run_id', 'skill_name', and 'cycle' to calculate the mean reward.

```python
# Group by 'run_id', 'skill_name', and 'cycle' to calculate mean reward
df_group = df_data.groupby(['run_id', 'skill_name', 'cycle'])['reward'].mean()
```

#### **b. Processing Observation Data**

We process the 'obs' column to create a DataFrame suitable for plotting.

```python
# Create a DataFrame from 'obs' dictionaries
obs_list = []
for obs in df_data['obs'].dropna():
    obs_list.append({k: v[0] if isinstance(v, list) else v for k, v in obs.items()})

df_obs = pd.DataFrame(obs_list)
df_obs['cycle'] = df_data['cycle'].values
df_obs['run_id'] = df_data['run_id'].values
df_obs['skill_name'] = df_data['skill_name'].values
```

#### **c. Visualizing Mean Episode Rewards**

We plot the mean episode rewards for each run and skill.

```python
# Plot Mean Episode Reward by Run ID and Skill
for run_id in df_group.index.get_level_values('run_id').unique():
    for skill in df_group.loc[run_id].index.get_level_values('skill_name').unique():
        mean_rewards = df_group.loc[run_id, skill]
        plt.figure(figsize=(10, 5))
        plt.plot(mean_rewards.index, mean_rewards.values)
        plt.ylabel('Mean Episode Reward')
        plt.xlabel('Cycle')
        plt.title(f'Run ID: {run_id} - Skill: {skill}')
        plt.grid(True)
        plt.show()
```

#### **d. Plotting Observation Data**

We plot the observation data to analyze the agent's state during training.

```python
# Plot Observation Data by Run ID and Skill
for run_id in df_obs['run_id'].unique():
    for skill in df_obs['skill_name'].unique():
        df_plot = df_obs[(df_obs['run_id'] == run_id) & (df_obs['skill_name'] == skill)]
        if not df_plot.empty:
            df_plot.set_index('cycle', inplace=True)
            df_plot.drop(['run_id', 'skill_name'], axis=1, inplace=True)
            df_plot.plot(subplots=True, figsize=(12, 8), title=f'Run ID: {run_id} - Skill: {skill}')
            plt.show()
```

There is much more for you to explore in the file as well.

#### **Explanation of Code Snippets**

- **Filtering Data**: We start by filtering the DataFrame `df` to include only relevant `category_sub` entries such as 'step', 'skill-training', and 'skill-training-cycle'.

- **Converting Data**: The 'data' column contains JSON strings. We use the `convert_to_dict` function to parse these strings into dictionaries.

- **Extracting Fields**: We extract the 'skill_name', 'reward', 'obs', and 'cycle' fields from the 'data' dictionaries. We use forward fill (`fillna(method='bfill')`) to propagate non-null values forward.

- **Processing Observations**: We process the 'obs' data, which contains observations from the agent's environment. We extract the first element from lists to get scalar values.

- **Grouping Data**: We group the data by 'run_id', 'skill_name', and 'cycle' to compute mean rewards, which we then plot over cycles.

- **Visualization**: We create plots to visualize mean episode rewards and observation data, which helps in analyzing the agent's performance over time.

---

## Additional Resources

- **Composabl Documentation**: Refer to the official Composabl documentation for more details on the data formats and available APIs.
- **Pandas Documentation**: [https://pandas.pydata.org/docs/](https://pandas.pydata.org/docs/)
- **Matplotlib Documentation**: [https://matplotlib.org/](https://matplotlib.org/)
- **Jupyter Notebook Documentation**: [https://jupyter.org/documentation](https://jupyter.org/documentation)

---

License

This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to contribute to this repository by submitting issues or pull requests. Your feedback is valuable in improving these resources!
