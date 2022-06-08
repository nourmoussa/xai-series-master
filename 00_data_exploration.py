# %% Imports
import matplotlib.pyplot as plt
from utils import DataLoader
import seaborn as sns
import numpy as np
import nbformat

# %% Load data
data_loader = DataLoader()
data_loader.load_dataset()
data = data_loader.data

# %% Show head
print(data.shape)
data.head()

# %% Show general statistics
data.info()

# %% Show histogram for all columns
columns = data.columns
for col in columns:
    print("col: ", col)
    data[col].hist()
    plt.show()

# %% Show preprocessed dataframe
data_loader.preprocess_data()
data_loader.data.head()


# %%
data.describe()

# %%
features = ['age', 'gender', 'height', 'weight', 'smoke', 'alco', 'active', 'cardio']

plt.figure(figsize=(8,8))

# Use an easier to see colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Mask
correlation = data[features].corr()
correlation[np.abs(correlation)<.05] = 0

sns.heatmap(correlation, annot = True, cmap=cmap).set(title = 'Correlation Heatmap')
plt.show()

# %%
df_clean = data.copy(deep=True)
# %%
df_clean['age'] = (df_clean['age']).round().astype('int')
# %%
df_clean.describe()
# %%
df_clean.agg({'age':['min','max','median','skew','std'],'gender':['min','max','median','skew','std'],'height':['min','max','median','skew','std']})
# %%
df_clean.groupby("smoke").mean()
# %%
df_clean.groupby("alco").mean()
# %%
df_clean.groupby("active").mean()
# %%
import plotly.express as px

df_sub2 = df_clean[['age', 'cholesterol','height','ap_hi', 'ap_lo','weight', 'cardio',]]
#Normalizing the data
df_normalized2 = (df_sub2-df_sub2.mean())/(df_sub2.std())
df_normalized2.cardio = df_sub2.cardio
df_normalized2.cholesterol = df_normalized2.cholesterol+np.random.rand(*df_normalized2.cholesterol.shape)/2
df_normalized2.ap_hi = df_normalized2.ap_hi+np.random.rand(*df_normalized2.ap_hi.shape)/2

fig = px.parallel_coordinates(df_sub2, color='cardio', labels={"age": "Age", "cholesterol": "Cholesterol","height": "Height", "ap_hi": "High Blood Pressure Value",
               "ap_lo": "Low Blood Pressure Value", "weight": "Weight", "cardio": "Cardiovascular Disease"},
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=1)
fig.show()
# %%
import plotly.express as px

df_sub2 = df_clean[['age', 'weight','height', 'smoke', 'alco', 'active', 'cardio']]
#Normalizing the data
df_normalized2 = (df_sub2-df_sub2.mean())/(df_sub2.std())
df_normalized2.cardio = df_sub2.cardio
#df_normalized2.cholesterol = df_normalized2.cholesterol+np.random.rand(*df_normalized2.cholesterol.shape)/2
#df_normalized2.ap_hi = df_normalized2.ap_hi+np.random.rand(*df_normalized2.ap_hi.shape)/2

fig = px.parallel_coordinates(df_sub2, color='cardio', labels={"age": "Age", "weight": "Weight","height": "Height", "smoke": "Smoke", "alco": "Alcohol consumption","active": "Active", "cardio": "Cardiovascular Disease"},
                             color_continuous_scale=px.colors.diverging.Tealrose,
                             color_continuous_midpoint=1)
fig.show()

# %%
# from pandas.plotting import parallel_coordinates


# #df_sub2 = df_clean[['age', 'weight', 'cholesterol', 'ap_hi', 'cardio', 'ap_lo']]

# df_sub = df_clean[['age', 'weight', 'cholesterol', 'ap_hi', 'ap_lo', 'cardio', 'height']].copy()
# df_sub.cardio = df_sub.cardio=='1' 
# #normalizing values
# df_normalized = (df_sub-df_sub.mean())/(df_sub.std())
# df_normalized.cardio = df_sub.cardio
# df_normalized.cholesterol = df_normalized.cholesterol+np.random.rand(*df_normalized.cholesterol.shape)/2
# df_normalized.ap_hi = df_normalized.ap_hi+np.random.rand(*df_normalized.ap_hi.shape)/2


# parallel_coordinates(df_normalized,'cardio')

# plt.show()
# %%
