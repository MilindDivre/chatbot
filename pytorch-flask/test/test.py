import requests
dataa = {"text": "hello"}
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
print(dataa)
resp = requests.post('http://localhost:8055/predict',json = dataa,headers = headers)
print(resp.text)