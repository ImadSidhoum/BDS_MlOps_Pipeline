import requests
import json

data = json.dumps({"signature_name": "serving_default", "name": "code.zip"})
headers = {"content-type": "application/json"}
json_response = requests.post(f'http://127.0.0.1:5001/update', data=data, headers=headers)
#js = json.loads(json_response.text)
print(json_response)