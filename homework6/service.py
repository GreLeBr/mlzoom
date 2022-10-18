import numpy as np
import pandas as pd
import bentoml
from bentoml.io import NumpyNdarray
from bentoml.io import JSON
from pydantic import BaseModel

import typing
from typing import TYPE_CHECKING
from typing import Any

model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")
# model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")
# dv = model_ref.custom_objects['dictVectorizer']

model_runner = model_ref.to_runner()
# dv = model_ref.custom_objects['dictVectorizer']

svc = bentoml.Service("mlzoomcamp_homework_pydantic", runners=[model_runner])

# class UserProfile(BaseModel):
#   name : str
#   age: int
#   country: str
#   rating: float
  
# input_spec = JSON(pydantic_model=UserProfile)

# @svc.api(input=input_spec ,output=JSON())
@svc.api(input=NumpyNdarray() , output=NumpyNdarray())
async def classify(application_data):
    # vector = dv.transform(application_data)
    # if application_data.request_id is not None:
    #     print("Received request ID: ", application_data.request_id)

    # input_df = pd.DataFrame([application_data])
    prediction = await model_runner.predict.async_run(application_data)
    # print(prediction)
    # result = prediction[0]
    # return {"prediction":"prediction", "result":"result"}
    return prediction[0]