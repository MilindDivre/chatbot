from flask_cors import CORS
from flask import Flask,jsonify,request
from torch_utils import get_prediction


app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})

@app.route('/predict',methods=['POST'])
def predict():
  if request.method == 'POST':
    content = request.get_json(silent=True)
    # print(content)
    sentence = request.json['text'] # instead of file we need to get sentence
    print("Sentence#############################",sentence)
    # if sentence is NONE or sentence == "":
    #   return jsonify({'error':'no sentence to inference'})
    # try:
    response = get_prediction(sentence)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Response",response)
    # except:
      # return jsonify({'error':'Error during prediction'})
    # response.headers.add("Access-Control-Allow-Origin", "*")
    return jsonify(ok=True, message=response)
  # return jsonify({'result': 1})

if __name__ == "__main__":
  app.run(debug=True,port=8055) 


