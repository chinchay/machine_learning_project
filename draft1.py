#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
# load data
df = pd.read_csv("assignment.csv")
df.head()
# %%
# inspect data
# no_nulls = set( df.isnull().mean() == 0 )
# df.loc[ df['par_5'].isnull() == True ] 
#%%
df2 = df.dropna()
df2.isnull().mean() == 0

# %%
# Reset index after drop
df = df.dropna().reset_index(drop=True)

#%%
df = df.drop( "Unnamed: 0", axis=1 )
# %%
# correlations
corr_matrix = df.corr()
corr_matrix['y'].sort_values(ascending=False)
# %%
from pandas.plotting import scatter_matrix

col_names = ["par_4", "par_12", "par_8", "par_16", "y"]
scatter_matrix(df[col_names], figsize=(12, 8))

# %%
from sklearn import preprocessing

# standarization (because it is know that we have "24 correlated columns")
# otherwise we would have used Min-max normalization
std_scaler = preprocessing.StandardScaler()
df2 = std_scaler.fit_transform( df.to_numpy() )
df2 = pd.DataFrame(df2, columns=list(df.columns))
df2.head()
df = df2

# %%
corr_matrix = df.corr()
corr_matrix['y'].sort_values(ascending=False)

# %%
# df.plot(x="par_0", y="y", kind="scatter")

from pandas.plotting import scatter_matrix
col_names = ["par_4", "par_12", "par_8", "par_16", "y"]
scatter_matrix(df[col_names], figsize=(12, 8), alpha=0.1)
# %%


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2 = scaler.fit_transform( df.to_numpy() )
# df2 = pd.DataFrame(df2, columns=list(df.columns))
# df2.head()

# from pandas.plotting import scatter_matrix
# col_names = ["par_4", "par_12", "par_8", "par_16", "y"]
# scatter_matrix(df2[col_names], figsize=(12, 8), alpha=0.1)
# # %%

# %%
# %%
# load data
df = pd.read_csv("assignment.csv")
df.head()

# Reset index after drop
df = df.dropna().reset_index(drop=True)

df = df.drop( "Unnamed: 0", axis=1 )

scaler = preprocessing.MinMaxScaler().fit(df.to_numpy())
print(scaler.data_max_)

col_names = list(df.columns)
from sklearn.preprocessing import MinMaxScaler
minmaxScaler = MinMaxScaler()
df = minmaxScaler.fit_transform( df.to_numpy() )
df = pd.DataFrame(df, columns=col_names)
df.head()

from pandas.plotting import scatter_matrix
col_names = ["par_4", "par_12", "par_8", "par_16", "y"]
scatter_matrix(df[col_names], figsize=(12, 8), alpha=0.1)


# %%
x = df["par_4"].to_numpy()
y = np.log(df["y"].to_numpy())
# %%
corr_matrix = df.corr()
corr_matrix['y'].sort_values(ascending=False)

# %%
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#     s=housing["population"]/100, label="population", figsize=(10,7),
#     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
# )
# plt.legendfrom pandas.plotting import scatter_matrix

col_names = ["par_4", "par_12", "par_8", "par_16", "y"]
scatter_matrix(df[col_names], figsize=(12, 8), alpha=0.1)
# %%
col_names = list(df.columns)
scatter_matrix(df[col_names], figsize=(12, 8), alpha=0.1)

# %%
corr_matrix = df.corr()
corr_matrix['y'].sort_values(ascending=False)

# %%
# df2 = df[["par_8", "par_16", "par_20", "par_4", "par_12", "par_0", "y"]]
# df2.head()
# # %%
# corr_matrix = df2.corr()
# corr_matrix['y'].sort_values(ascending=False)
# # %%

# lista = [ "par_4", "par_8", "y"]
# lista = [ "par_4", "par_8", "par_12", "y"]
lista = [  "par_8", "par_16",   "par_4", "par_12", "par_0", "y"]
# df2 = df[lista]
df2 = df[lista]
n = len(lista) - 1
df2.head()
# %%
corr_matrix = df2.corr()
corr_matrix['y'].sort_values(ascending=False)
# %%
from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(n, interaction_only=True)
poly = PolynomialFeatures(n)
X = df2.to_numpy()
# X = X[:, :-1]
X = X[:, :n]
Y = df2["y"].to_numpy()
Xpoly = poly.fit_transform(X)

# %%
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(Xpoly, Y)
# lin_reg.fit(X, Y)
# %%
plt.plot(lin_reg.coef_, ".")

# %%
# np.around(lin_reg.coef_[:10], decimals=3)

# %%
pred = lin_reg.predict(Xpoly)
plt.plot(Y, pred, ".")
# %%

corr_matrix = abs( df.corr() )
corr_matrix['y'].sort_values(ascending=False)
# %%
