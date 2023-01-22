#python -m uvicorn main:app --host 0.0.0.0 --port 10000

import pandas as pd
import json
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
train = joblib.load('train.sav')
train_combined = joblib.load('train_combined.sav')
anime = pd.read_parquet('final_anime_list.parquet')
K_model_cos = joblib.load('K_model_cos.sav')
K_combined_cos = joblib.load('K_combined_cos.sav')


train_tv = joblib.load('train_tvseries.sav')
tvseries = pd.read_parquet('tvseries.parquet')
K_model_tf = joblib.load('K_model_tvseries.sav')

train_movies = joblib.load('train_movies.sav')
movies = pd.read_parquet('moviesfile.parquet')
movies_model = joblib.load('model_movies.sav')

class title(BaseModel):
    names:str

class recom(title):
    recommendations:dict

@app.post('/anime')
def recommend(input:title):
    input = input.names
    input = (input.lower()).split('|')

    if len(input)==1:           
        sample = train.toarray()[anime.index[anime['English'].str.lower().str.contains(input[0])]][0]
        distances_cos, indices_cos = K_model_cos.kneighbors([sample])
        results = anime.iloc[indices_cos.squeeze()[0:]]
        results.drop(['Japanese'], axis=1, inplace=True)
        results['Distances'] = distances_cos.squeeze()[0:]
        return json.loads(results.to_json(orient='records'))
    
    else:
        samples=[]
        for name in input:
            samples.append(train_combined.toarray()[anime.index[anime['English'].str.lower().str.contains(name)]][0])
        sample_mean = [sum(sub_list) / len(sub_list) for sub_list in zip(*samples)]
        distances_cos_comb, indices_cos_comb = K_combined_cos.kneighbors([sample_mean])
        results = anime.iloc[indices_cos_comb.squeeze()[0:]]
        results.drop(['Japanese'], axis=1, inplace=True)
        results['Distances'] = distances_cos_comb.squeeze()[0:]
        return json.loads(results.to_json(orient='records'))



@app.post('/movies')
def recommend(input:title):
    input = input.names
    input = (input.lower()).split('|')
        
    
    samples=[]
    for name in input:
        samples.append(train_movies.toarray()[movies.index[movies['title'].str.lower().str.contains(name)]][0])
    sample_mean = [sum(sub_list) / len(sub_list) for sub_list in zip(*samples)]
    distances_movies, indices_movies = movies_model.kneighbors([sample_mean])
    results = movies.iloc[indices_movies.squeeze()]
    results['Distances'] = distances_movies.squeeze()
    return json.loads(results.to_json(orient='records'))


@app.post('/tvseries')
def recommend(input:title): 
    input=input.names
    sample = train_tv.toarray()[tvseries.index[tvseries['Series Title'].str.lower().str.contains(input)]][0]
    distances_cos_comb, indices_cos_comb = K_model_tf.kneighbors([sample])
    results = tvseries.iloc[indices_cos_comb.squeeze()[0:]]
    results['Distances'] = distances_cos_comb.squeeze()[0:]
    return json.loads(results.to_json(orient='records'))
