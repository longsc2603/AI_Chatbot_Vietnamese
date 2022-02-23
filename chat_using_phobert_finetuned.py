import json
import torch
import random
from test_phobert_finetuned import tags_set, model, tokenizer

with open('test_content.json', 'r') as json_data:
    contents = json.load(json_data)


def chat_bot_PhoBERT(sentence):
    token = tokenizer(sentence, max_length=13, padding='max_length',
                      truncation=True)
    X_mask = torch.tensor(token['attention_mask'])
    X = torch.tensor(token['input_ids'])
    with torch.no_grad():
        preds = model(X, X_mask)
    preds = torch.argmax(preds, dim=1)
    tag = tags_set[preds.item()]
    for content in contents['intents']:
        if tag == content['tag']:
            answer = random.choice(content['responses'])
    return answer
