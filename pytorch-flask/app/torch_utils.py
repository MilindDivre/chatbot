import random
import json
import os
import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
json_url = os.path.join(SITE_ROOT, "static", "intents.json")
# with open('intents.json', 'r') as json_data:
intents = json.load(open(json_url))

model_url=os.path.join(SITE_ROOT, "static", "data.pth")
# FILE = "data.pth"
data = torch.load(model_url)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def get_prediction(sentence):
  # bot_name = "Sam"
  # print("Let's chat! (type 'quit' to exit)")
  # while True:
  #     # sentence = "do you use credit cards?"
  #     sentence = input("You: ")
  #     if sentence == "quit":
  #         break
  print("#######################################get prediction called",sentence)
  sentence = tokenize(sentence)
  X = bag_of_words(sentence, all_words)
  X = X.reshape(1, X.shape[0])
  X = torch.from_numpy(X).to(device)

  output = model(X)
  _, predicted = torch.max(output, dim=1)

  tag = tags[predicted.item()]

  probs = torch.softmax(output, dim=1)
  prob = probs[0][predicted.item()]
  if prob.item() > 0.75:
      for intent in intents['intents']:
          if tag == intent["tag"]:
            response = random.choice(intent['responses'])
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$",response)
            return (response)
  else:
      return ("{bot_name}: I do not understand...")