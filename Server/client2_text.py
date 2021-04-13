import requests
import json

data = json.dumps({"signature_name": "serving_default", "name": "04-12-21_18-35-49.zip", "version": "1"})
headers = {"content-type": "application/json"}
json_response = requests.post(f'http://127.0.0.1:5002/update', data=data, headers=headers)
js = json.loads(json_response.text)
#json_response= requests.get(f'http://127.0.0.1:5001/version')
#print(type(json_response.text))