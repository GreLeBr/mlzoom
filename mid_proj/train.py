import pickle
import bentoml
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold , train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# parameters

n_splits = 5
output_file = f'ranforeg.bin'


# data preparation

# data = pd.read_csv('./data/data.parquet')

data = pd.read_parquet('./data/data.parquet')

df_full_train, df_test = train_test_split(data, test_size=0.2, random_state=1)

kept_cols = ["bird_distance","emplacement_pk_start",	"emplacement_pk_end"]


# training 

def train(df_train, y_train):
    dicts = df_train[kept_cols].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = RandomForestRegressor(n_estimators=30, max_depth=20, 
                           random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)
    
    return dv, model


def predict(df, dv, model):
    dicts = df[kept_cols].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict(X)

    return y_pred


# validation

print(f'doing validation')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = np.log1p(df_train.duration_sec.values)
    y_val = np.log1p(df_val.duration_sec.values)

    dv, model = train(df_train, y_train)
    y_pred = predict(df_val[kept_cols], dv, model)

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    scores.append(rmse)

    print(f'rmse {fold} is {rmse}')
    fold = fold + 1


print('validation results:')
print('{:3f} +- {:3f}'.format( np.mean(scores), np.std(scores)))


# training the final model

print ('training the final model')

dv, model = train(df_full_train, np.log1p(df_full_train.duration_sec.values))
y_pred = predict(df_test, dv, model)

y_test = np.log1p(df_test.duration_sec.values)
rmse = mean_squared_error(y_test, y_pred)

print(f'rmse={rmse}')


# Save the model

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')


bentoml.sklearn.save_model("trip_duration", model)