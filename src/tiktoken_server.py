'''
Rest service that exposes a single POST endpoint that accepts a JSON object 
with a  key, "prompt" and an optional key "model", and returns a JSON object with 
a single key, "tokens" # that has an array of the tokens returned by tiktoken.

Prompt is a string that needs to be tokenized. model is a string that is the name 
of a model in the  list of models returned by tiktoken.models(). If model is not
provided, use the default model "text-davinci-003". If the model is not found,
return an empty array. If the model is found, 
return the tokens returned by tiktoken.tokenize() for the prompt and model.

@Author: howdymic

'''


import json

from flask import Flask, request
import tiktoken

app = Flask(__name__)

@app.route("/chunks", methods=['POST', 'GET'])
def chunks():
    data = request.get_json()
    if not 'prompt' in data:
      return json.dumps({'tokens': []})
    model = "text-davinci-003"

    if 'chunk_size' in data:
        chunk_size = int(data['chunk_size'])
    else:
        chunk_size = 2000

    if 'overlap' in data:
        overlap = int(data['overlap'])
    else:
        overlap = 100

    if  'model' in data:
      model = data['model']

    try :
        print('Getting encoding for model' + model)
        enc = tiktoken.encoding_for_model(model)
    except :
        return json.dumps({'error': "Model not found"})

    text = data['prompt']
    tokens = enc.encode(text)
    num_tokens = len(tokens)

    chunks = []
    for i in range(0, num_tokens, chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(enc.decode(chunk))

    return json.dumps({
        'overlap': overlap,
        'chunk_size': chunk_size,
        'chunks': chunks
    })


@app.route("/tokenize", methods=['POST', 'GET'])
def tokenize():
    data = request.get_json()
    if not 'prompt' in data:
      return json.dumps({'tokens': []})
    model = "text-davinci-003"

    if  'model' in data:
      model = data['model']

    print('Model:', model)
    print('Data:' , data)
    try :
        print('Getting encoding for model' + model)
        enc = tiktoken.encoding_for_model(model)
    except :
        return json.dumps({'error': "Model not found"})
    tokens = enc.encode(data['prompt'])
    return json.dumps({'tokens': tokens, 'count': len(tokens)})

@app.route("/token_count", methods=['POST', 'GET'])
def token_count():
    data = request.get_json()
    if not 'prompt' in data:
      return json.dumps({'tokens': []})
    model = "text-davinci-003"

    if  'model' in data:
      model = data['model']

    print('Model:', model)
    print('Data:' , data)
    try :
        print('Getting encoding for model' + model)
        enc = tiktoken.encoding_for_model(model)
    except :
        return json.dumps({'error': "Model not found"})
    tokens = enc.encode(data['prompt'])
    return json.dumps({'count': len(tokens)})


if __name__ == "__main__":
    app.run(host='0.0.0.0')


# python -m flask --app test.py run
#  curl -d '{"prompt" : "hello world", "model" : "text-davinci-003"}' -H "Content-Type: application/json"  http://localhost:5000/tokenize
