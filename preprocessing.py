import pandas as pd
#Pipline Imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder


def run_X_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    '''Processes the X dataframe so that all the values are Numeric and model ready'''
    age_pipeline = Pipeline([('scaler', MinMaxScaler())])
    sex_pipeline = Pipeline([('ohe', OneHotEncoder(sparse_output=False, drop='first'))])
    loc_pipeline = Pipeline([('lbl_E', LabelEncoder())])
    preprocessor = ColumnTransformer([
        ('age', age_pipeline, ['age']),
        ('sex', sex_pipeline, ['sex']),
        ('loc', loc_pipeline, ['localization'])
    ])
    df = preprocessor.fit_transform(df)
    return df

def run_y_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    '''Processes the y dataframe so that all the values are Numeric and model ready'''
    y_pipeline = Pipeline([('ohe', OneHotEncoder(sparse_output=False, drop=None))])
    df = y_pipeline.fit_transform(df)
    return df

def preprocess_metadata(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''preprocess's metadata from the skin-cancer-mnist-ham10000 dataset
    returns a preprocessed version of X and y'''
    df = df.drop(columns=[col for col in ['dx_type','lesion_id'] if col in df.columns])
    # fill age with mean values
    df['age'] = df['age'].fillna((df['age'].mean()))
    # Drop the unknown sex names
    df = df[df['sex'] != 'unknown']
    #drop unknowns
    df = df[df['localization'] != 'unknown']
    # return processed df
    y = df['dx']
    X = df.drop(columns=['dx'])
    X = run_X_pipeline(X)
    y = run_y_pipeline(y)
    return X, y

