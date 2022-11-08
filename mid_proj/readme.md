# Predicting of trip duration for a ride between two Bixi stations

I ran out of time and did not find a good dataset I like so I tried with this one but it is a tricky one. 

The data I used is the open data regarding the bike sharing service **Bixi** available in Montréal Québec Canada. 

The data I used as well as more is available at [Bixi open data](https://bixi.com/fr/donnees-ouvertes]


1. Goal of the project:

   The goal of this project is to predict the duration of a trip knowing departing and arrival stations. 

2. Limitations:

   This type of prediction is very simple and a classic for ride precitions, however this dataset is rather ill fitted for the proposed goal. 
   Indeed it cannot be assumed that someone borrowing a bike has necessarily the intention to get from point A to point B  
   Testing whether it was possible to predict how long a bike might be borrowed without knowing where it is parked back did not lead to any much better result that a dummy prediction. 
   Trying to predict which station a bike might be return back depending on where it was taken would be even more interesting, potentially opening the door of optimizing bike availibilty over the day. However it is a much more difficult problem out of the scope of this mini project.  

3. EDA:

   The data is pretty cleaned and not missing values. There is only one station for which localization data was bad and related data was removed. 
   In order to get more relevant data to predict trip duration from, I limited entries to trips happening during the morning or evening rush, when, although it is my personal presumption, trips are more likely to be motivated by an efficient commute. For the same reason, only trips further than 1km away were considered.   

4. Features engineering:

  A very simple calculation of the distance between both station was made using an haversine function. 
  A formulat to estimate shortest path by bike is in the notebook but it is not scaled to be executed on a high number of rows an I might try to incoporate it later. 
  Using information of when a ride was initiated and decomposing it by hour of the day or seeing if the day of the week is impactful as people are less likely to commute over the week-end did not improve prediction. 
  Since there are many combinations of departure-arrival stations, its high cardinality prevents it from being useful or useable. 
  Using a shapefile of neighborhood in Montréal, I assigned the neighborhood to the departing station but this feature did not seem to have mnuch of any impacts either and was ignored. [shapefile](https://www.donneesquebec.ca/recherche/dataset/vmtl-quartiers)
  Finally I downloaded historical weather data checking whether it is raining or not during the day the bike is borrowed or locked back. This feature did not help prediction either. Data is available here : [weather historical data](https://climate.weather.gc.ca/historical_data/search_historic_data_stations_e.html?StationID=48374&Month=8&Day=1&Year=2022&timeframe=3&StartYear=1840&EndYear=2020&type=bar&MeasTypeID=totprecip&wbdisable=true&searchType=stnProx&txtRadius=25&optProxType=navLink&txtLatDecDeg=45.466666666667&txtLongDecDeg=73.75&optLimit=specDate&selRowPerPage=25&station=MONTREAL%2FPIERRE+ELLIOTT+TRUDEAU+INTL+A#wb-cont)

5. Training:

   I checked a variety of models, in the end the "best" results seems to be with a RandomForestRegressor with xgboost close behind. However most of the models were performing more or less the same and no hyperparameter tuning could improve the rather lackluster predictive accuracy of the model. 
   The pickle file is too big to be uploaded to github basic so you will need to run the train.py script. (probably should be simplified on my hand)

6. Running locally: 

   (You will need to build the model first: run `python train.py` or `pipenv run python train.py` 
   You can either your preinstalled libraries or install/use `pipenv install`
   To run the api do `bentoml serve service.py:svc` , if you running a pipenv environment then `pipenv run bentoml serve service.py:svc` 
   Open the api website (usually http://localhost:3000/  or http://0.0.0.0:3000/  (if on windows?))
   You need to put a number for the departing and the arriving station, the app will fail if the number is not in the list of available stations, sadly you will have to explore the file stations.parquet to see the numbers or try : *depart_station=1215, arrival_station=1174* 

6. Bentos / docker

   To build an image with bentoml you need to create a bento (aka instructions to build a docker). For that you need to create a 
   `touch bentofile.yaml` instructions to build the file are available in [BentoMl documentation](https://docs.bentoml.org/en/latest/concepts/bento.html) , the python packages will be the same as the one used for pipenv minus bentoml. 

   don't forget to add `- "data/"` in the include section in order to get required files
   Once the bentofile has been built, you need to go to the folder where it is saved on your computer to run the bento. 
   Instructions to deploy locally can be found in the [documentation](https://docs.bentoml.org/en/latest/concepts/deploy.html)

7. Heroku deployment

   For the few days it lasts before heroku stops being free (28 november 2022), you can deploy your bento on heroku if you have an account: 
   Move to the folder with the ben you want to deploy and go inside the docker folder
   ```
   heroku login
   heroku container:login
   heroku create yourappname
   heroku container:push web --app yourappname --context-path=../..
   heroku container:release web --app pet-pawpularity
    ```
    Example of the current api deploy at [my deployment](https://montreal-bikes.herokuapp.com/)
    
    I might do frontend later if I have time sorry
