import numpy as np
import pandas as pd
import bentoml
import math
# from bentoml.io import NumpyNdarray
from bentoml.io import JSON
from pydantic import BaseModel, validator

# import typing
from typing import TYPE_CHECKING
from typing import List

stations = pd.read_parquet("./data/stations.parquet")
model_ref = bentoml.sklearn.get("trip_duration:latest")
# model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
# dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()
# dv = model_ref.custom_objects['dictVectorizer']

svc = bentoml.Service("trip_duration", runners=[model_runner])




class UserProfile(BaseModel):
  depart_station : int  
  arrival_station: int
  @validator("depart_station", "arrival_station")
  @classmethod  # Optional, but your linter may like it.
  def check_station(cls, value):
      if value not in stations.pk.unique().tolist():
          raise ValueError("Not a valid station number")
      return value

input_spec = JSON(pydantic_model=UserProfile)

def distance(origin, destination):
    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : tuple of float
        (lat, long)
    destination : tuple of float
        (lat, long)

    Returns
    -------
    distance_in_km : float

    Examples
    --------
    >>> origin = (48.1372, 11.5756)  # Munich
    >>> destination = (52.5186, 13.4083)  # Berlin
    >>> round(distance(origin, destination), 1)
    504.2
    """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

def prepare_features(trip):
    
    depart_station , arrival_station = trip    
    features = {}
    origin = stations[stations["pk"]==int(depart_station[1]) ][["latitude", "longitude"]].values[0]
    destination = stations[stations["pk"]==int(arrival_station[1])][["latitude", "longitude"]].values[0]

    features["bird_distance"] = distance(origin, destination)
    features["emplacement_pk_start"]= depart_station[1]
    features["emplacement_pk_end"] = arrival_station[1]
    
    return features




@svc.api(input=input_spec ,output=JSON())
# @svc.api(input=NumpyNdarray() , output=NumpyNdarray())

async def classify(application_data):
    collected_values = application_data
    # vector = dv.transform(application_data)
    # if application_data.request_id is not None:
    #     print("Received request ID: ", application_data.request_id)

    # input_df = pd.DataFrame([application_data])
    
    features = prepare_features(collected_values)
    X = pd.DataFrame([features])
    prediction = await model_runner.predict.async_run(X)
    # print(prediction)
    # result = prediction[0]
    # return {"prediction":"prediction", "result":"result"}
    return np.expm1(prediction[0])