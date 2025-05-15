import numpy as np
import torch
import tiktoken
from Model import *

# hyperparameters
d_model = 512
num_heads = 8
context_len = 32
batch_size = 4
dropout = 0.1
max_iter = 50000
learning_rate = 1e-4
train_evaluate_range = 20
valid_evaluate_range = 50
valid_iter = 10
# epoch = 20
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('sales_textbook.txt'):
    text = open('sales_textbook.txt').read()

tokenizer = tiktoken.get_encoding("o200k_base")
tokenizer_text = tokenizer.encode(text)
tokenizer_text = torch.tensor(tokenizer_text, dtype=torch.long, device=device)
max_token_value = len(tokenizer_text)
print(max_token_value)
data_split_size = 0.9
train_dataset = tokenizer_text[:int(len(tokenizer_text) * data_split_size)]
valid_dataset = tokenizer_text[int(len(tokenizer_text) * data_split_size):]


def get_batch(mode):
    if mode == 'train':
        dataset = train_dataset
    elif mode == 'valid':
        dataset = valid_dataset

    idxs = torch.randint(low=0, high=len(dataset) - context_len, size=(batch_size,))
    x = torch.stack([dataset[idx:idx + context_len] for idx in idxs]).to(device)
    y = torch.stack([dataset[idx + 1:idx + 1 + context_len] for idx in idxs]).to(device)

    return x, y


model = Model().to(device)
print(sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
train_losses = []


print('dataset loading success')

for step in range(max_iter):

    x, y = get_batch('train')
    _, loss = model(x, y)

    train_losses.append(loss.item())
    if step != 0 and step % train_evaluate_range == 0:
        losses = np.mean(train_losses[step - train_evaluate_range:step])
        # print(train_losses)
        print('step', step, 'avg_train_loss', round(losses, 4))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step != 0 and step % valid_evaluate_range == 0:
            valid_losses = []
            model.eval()
            with torch.no_grad():
                for _ in range(valid_iter):
                    x, y = get_batch('valid')
                    _, loss = model(x, y)
                    # print('valid_loss', round(loss.item(), 4))
                    valid_losses.append(loss.item())
                losses = np.mean(valid_losses)
                print('******************************************')
                print('step', step, 'avg_valid_loss', round(losses, 4))
                print('******************************************')

                # predict_context = model.generate(x)
                #
                # print(tokenizer.decode((predict_context[0].tolist())))

            model.train()

torch.save(model.state_dict(), 'model_1.ckpt')

