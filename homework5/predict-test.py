#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://0.0.0.0:9696/predict'

client_id = 'client-test'
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}
client = {"reports": 0, "share": 0.245, "expenditure": 3.438, "owner": "yes"}

response = requests.post(url, json=client).json()
print(response)

if response['price'] == True:
    print(f'A credit card for {client_id}')
else:
    print(f'No credit card for {client_id}, thank you algorithm')   