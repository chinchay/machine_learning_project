#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures
#
class Preparer(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        super().__init__()
    #
    def fit(self, df):
        return self
    #
    def transform(self, df):
        # Drop rows with containing missing data
        # and reset index after drop
        df = df.dropna().reset_index(drop=True)
        
        # Remove indexing column (it does not contribute to 'y')
        if 'Unnamed: 0' in list(df):
            df = df.drop( "Unnamed: 0", axis=1 )
            # df.head()
        #
        
        return df
        # return super().fit_transform(X, y, **fit_params)
    #
#

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
#
num_pipeline = Pipeline(
    [
        # ( 'imputer', SimpleImputer(strategy="mean") ),
        ( 'prep',    Preparer() ), # Here it will take care of NaN/missing cells
        ( 'scaler',  MinMaxScaler() ),
        # it will return a np.array, 
        # so don't worry abount saving `col_names`: col_names = list(df.columns) = list(df)
    ]
)
#

# def np2pandas(arr, col_names):
#     return pd.DataFrame(arr, columns=col_names)
# #

# For the model:
def getFeatLabels(arr):
    # X     = dframe.drop("y", axis=1)
    # y     = dframe["y"].copy()
    X = arr[:, :-1]
    y = arr[:, -1]
    return X, y
#


def getModel(modelName):
    if modelName == "linear":
        return LinearRegression()
    elif modelName == "randomForest":
        return RandomForestRegressor()
    #
#

def pipe_RMSE_model(model, modelName, X, y, listColumns=None):
    rows = 200
    if listColumns == None:
        listColumns = [i for i in range( X.shape[1] )]
    #
    
    # Training on all the rows, considering only the columns in `listColumns`
    model.fit( X[:, listColumns], y )
    
    # visualizing the coefficients
    if modelName == "linear":
        plt.plot(model.coef_, ".")
        plt.xlabel("i-th coeff")
        plt.ylabel("coeff value")
    #
    
    # Preparing a small "test set" from training set:
    _X = X[:rows, listColumns]
    _y = y[:rows]
    
    # Prediction on the small "test set":
    y_pred = model.predict( _X )
    
    # display
    print( "RMSE for " + modelName + ":\n ", getRMSE( _y, y_pred ) )
    
    return model
#

def plot_actual_vs_prediction(labels, y_pred):
    plt.plot( labels, y_pred, ".", alpha=0.4)
    plt.xlim( (0, 1) )
    plt.ylim( (0, 1) )
    plt.plot( [0, 0.5, 1], [0, 0.5, 1], 'k-' )
    plt.xlabel("actual values")
    plt.ylabel("prediction")
    plt.show()
#


from sklearn.metrics import mean_squared_error
#
def getRMSE(y_test, y_prediction):
    final_mse  = mean_squared_error( y_test, y_prediction )
    final_rmse = np.sqrt(final_mse)
    return final_rmse
#

def validation( df_test, trainedModel, listColumns=None, poly=None ):
    # Work on the test set
    
    Xy_test_tr = num_pipeline.fit_transform( df_test )

    X_test_tr, y_test_tr = getFeatLabels( Xy_test_tr )

    if listColumns != None:
        X_test_tr = X_test_tr[ :, listColumns ]

        if poly != None: # it must be here, in a reduced number of columns!!
            X_test_tr = poly.fit_transform( X_test_tr )
        #
    #

    # Evaluation on the transformed test set:
    y_pred = trainedModel.predict( X_test_tr )

    print( "RMSE: ", getRMSE(y_test_tr, y_pred) )

    plot_actual_vs_prediction(y_test_tr, y_pred)
#

    
#%%
# load data
df = pd.read_csv("assignment.csv")
df.head()

#%%
# Split data
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train.head()

#%%
# # plot data to obtain insights
# from pandas.plotting import scatter_matrix
# col_names = list(df_train)
# scatter_matrix(df_train[col_names], figsize=(24, 16), alpha=0.1)

#%%
# correlation matrix
corr_matrix = df_train.corr()
corr_matrix['y'].sort_values(ascending=False)

#%%
# sorting by absolute value
corr_matrix = abs( df_train.corr() )
corr_matrix['y'].sort_values(ascending=False)

#%%
corr =  abs( df_train.corr() )
corr.style.background_gradient(cmap='jet', vmax=0.32).set_precision(2)
# parameters are independents between them

#%%
# Data manipulation
Xy_train_tr = num_pipeline.fit_transform(df_train)

# Features and labels
X_train_tr, y_train_tr = getFeatLabels( Xy_train_tr )

Xcopy = X_train_tr.copy()

#%%
# Work with the best feature columns

# onlyCols = [ 8, 16,   4, 12 ]
onlyCols = [ 8, 16,   4, 12, 20 ]
# onlyCols = [ 8, 16,   4, 12, 20, 0 ]
n = len(onlyCols)

X_train_tr = X_train_tr[ :, onlyCols ]

#%%
#
# We will use a linear regression for 5 entries x1, x2, x3, x4, x5, 
# so we prepare the expand the input to the appropriate shape
from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(n, interaction_only=True)
poly = PolynomialFeatures(n)
# poly = PolynomialFeatures(1)
#
Xp_train_tr = poly.fit_transform( X_train_tr )
print(Xp_train_tr.shape)

#%%
# Building the model

modelName = "linear"
model     = getModel(modelName)
trainedL1 = pipe_RMSE_model( model, modelName, Xcopy, y_train_tr )

#%%
modelName = "linear"
model     = getModel(modelName)
trainedL2 = pipe_RMSE_model( model, modelName, Xcopy, y_train_tr, listColumns=onlyCols )

#%%
modelName = "linear"
model     = getModel(modelName)
trainedL3 = pipe_RMSE_model( model, modelName, Xp_train_tr, y_train_tr )

#%%
modelName = "randomForest"
model     = getModel(modelName)
trainedR1 = pipe_RMSE_model( model, modelName, Xcopy, y_train_tr )

#%%
modelName = "randomForest"
model     = getModel(modelName)
trainedR2 = pipe_RMSE_model( model, modelName, Xcopy, y_train_tr, listColumns=onlyCols )

#%%
validation( df_test, trainedL3, listColumns=onlyCols, poly=poly )

#%%
validation( df_test, trainedR2, listColumns=onlyCols, poly=None )






# #%%
# from sklearn.model_selection import GridSearchCV
# #
# param_grid = [
#     {'n_estimators': [3, 10, 30, 50, 100], 'max_features': [2, 4, 6, 8]},
#     {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
#   ]

# model = RandomForestRegressor()

# grid_search = GridSearchCV(model, param_grid, cv=5,
#                            scoring='neg_mean_squared_error',
#                            return_train_score=True)

# grid_search.fit( Xcopy[:, onlyCols], y_train_tr )
# #%%
# model = RandomForestRegressor(n_estimators=100, max_features=8)


