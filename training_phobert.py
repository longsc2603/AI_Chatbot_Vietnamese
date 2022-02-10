import torch
from transformers import AutoModel, AutoTokenizer, AdamW
from torch import nn as nn
import json
from torch.utils.data import DataLoader, TensorDataset
from phobert_finetuned import PhoBERT_finetuned

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load data
with open('content.json', 'r', encoding="utf-8") as c:
    contents = json.load(c)
# Load model PhoBERT and its tokenizer
phobert = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

tags = []
X = []
y = []

for content in contents['intents']:
    tag = content['tag']
    for pattern in content['patterns']:
        X.append(pattern)
        tags.append(tag)

tags_set = sorted(set(tags))

for tag in tags:
    label = tags_set.index(tag)
    y.append(label)
token_train = {}
token_train = tokenizer.batch_encode_plus(
    X,
    max_length=15,
    padding='max_length',
    truncation=True
)
X_train_mask = torch.tensor(token_train['attention_mask'])
X_train = torch.tensor(token_train['input_ids'])
y_train = torch.tensor(y)

# Hyperparameter:
batch_size = 8
hidden_size = 512
num_class = len(tags)
lr = 7.5e-5
num_epoch = 100

dataset = TensorDataset(X_train, X_train_mask, y_train)
train_data = DataLoader(dataset=dataset, batch_size=batch_size,
                        shuffle=True)

# Model:
for param in phobert.parameters():
    param.requires_grad = False
model = PhoBERT_finetuned(phobert, hidden_size=hidden_size,
                          num_class=num_class)
model = model.to(device)
optimizer = AdamW(model.parameters(), lr=lr)
loss_f = nn.NLLLoss()


def train():
    model.train()
    total_loss = 0
    for (step, batch) in enumerate(train_data):
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, labels = batch
        # clear previously calculated gradients
        model.zero_grad()
        pred = model(sent_id, mask)
        loss = loss_f(pred, labels)
        total_loss += loss.item()
        loss.backward()
        # clip the gradients to 1.0.
        # It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    avg_loss = total_loss/len(train_data)
    print(avg_loss)


for epoch in range(num_epoch):
    print('\n Epoch {:}/{:}'.format(epoch + 1, num_epoch))
    train()
torch.save(model.state_dict(), 'saved_weights.pth')
