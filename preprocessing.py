import pandas as pd
#Pipline Imports
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, LabelEncoder

def run_X_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    age_pipeline = Pipeline([('scaler', MinMaxScaler())])
    cat_pipeline = Pipeline([('ohe', OneHotEncoder(sparse_output=False, drop='first'))])
    preprocessor = ColumnTransformer([
    ('passthrough', 'passthrough', ['image_id']),
    ( 'age', age_pipeline, ['age']),
    ('cat', cat_pipeline,['sex','localization'])
    ])
    preprocessor.set_output(transform='pandas')
    df = preprocessor.fit_transform(df)
    return df

def run_y_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    '''Processes the y dataframe so that all the values are Numeric and model ready'''
    y_pipeline = Pipeline([('ohe', OneHotEncoder(sparse_output=False, drop=None))])
    y_pipeline.set_output(transform="pandas")
    y_pipeline.set_output(transform='pandas')
    df = y_pipeline.fit_transform(df)
    return df

def preprocess_metadata(df: pd.DataFrame, split=True):# -> tuple[pd.DataFrame, pd.DataFrame]:
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
    if split:
        y = df[['dx']]
        X = df.drop(columns=['dx'])
        X = run_X_pipeline(X)
        y = run_y_pipeline(y)
        return X, y
    else:
        return df

