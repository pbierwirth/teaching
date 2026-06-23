import numpy as np
import pandas as pd

#%%
# 1. Parameters extracted and reverse-engineered from the meta-analysis
n_control = 1052
n_depressed = 1354

# We need a plausible baseline for morning salivary cortisol in nmol/l.
# Let's set the control group to 15.0 nmol/l.
mean_control = 15.00
mean_depressed = 15.00 + 2.58 # The reported mean difference

# The reverse-engineered standard deviation
std_dev = 20.24 

# Set random seed so all students get the exact same "random" dataset
np.random.seed(42)

# 2. Simulate the data
# (Note: We use a normal distribution to match standard t-test math, 
# though biological cortisol is technically bounded at 0 and skewed).
cortisol_control = np.random.normal(loc=mean_control, scale=std_dev, size=n_control)
cortisol_depressed = np.random.normal(loc=mean_depressed, scale=std_dev, size=n_depressed)

# Optional: Clip negative values to make it biologically realistic (0.1 is the detection limit)
cortisol_control = np.clip(cortisol_control, a_min=0.1, a_max=None)
cortisol_depressed = np.clip(cortisol_depressed, a_min=0.1, a_max=None)

# 3. Build the Pandas DataFrame
df_control = pd.DataFrame({
    'Patient_ID': [f'C_{i:04d}' for i in range(1, n_control + 1)],
    'Group': 'Control',
    'Morning_Cortisol_nmol_L': np.round(cortisol_control, 2)
})

df_depressed = pd.DataFrame({
    'Patient_ID': [f'D_{i:04d}' for i in range(1, n_depressed + 1)],
    'Group': 'Depression',
    'Morning_Cortisol_nmol_L': np.round(cortisol_depressed, 2)
})

# Combine the groups and shuffle the rows so it looks like a raw dataset
df = pd.concat([df_control, df_depressed]).sample(frac=1).reset_index(drop=True)

# Export for the students (optional)
# df.to_csv("cortisol_study_data.csv", index=False)

print(df.head(10))

df.to_csv('cortisol_depressed_vs_healthy.csv', index=False)

#%% calculating AUC
import matplotlib.pyplot as plt
df = pd.read_csv('cortisol_depressed_vs_healthy.csv')
sort_idx = np.argsort(df.Morning_Cortisol_nmol_L)[::-1]


boolean = df.Group == 'Depression'

sensitivity = np.cumsum(boolean[sort_idx])/sum(boolean)
fp = np.cumsum(~boolean[sort_idx])/sum(~boolean)

plt.plot(fp, sensitivity)
plt.plot([0,1],[0,1])


sum(np.diff(fp)*sensitivity[1:])
