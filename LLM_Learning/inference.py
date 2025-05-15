import tiktoken


from Model import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model().to(device)

checkpoint = torch.load('model.ckpt', map_location=torch.device('cuda'))

model.load_state_dict(checkpoint)

model.eval()

tokenizer = tiktoken.get_encoding('o200k_base')
batch_size = 1

inputs = 'In this chapter, we will delve into the various techniques that sales professionals can employ to capture the attention of their potential customers. By using attention-grabbing techniques effectively, you will be able to create a strong initial impression and pique the curiosity of your audience, thus increasing the likelihood of a successful sales interaction.'
inputs_token = torch.tensor(tokenizer.encode(inputs), dtype=torch.long).to(device).unsqueeze(0)
# print(len(inputs_token.tolist()))

# if len(inputs_token.tolist()) <= context_len:
#     padding = torch.ones(batch_size, context_len - len(inputs_token) + 1, dtype=torch.long).to(device)
#
#     print(padding.shape)
#     inputs_token = torch.cat([padding, inputs_token], dim=-1)z`
#
#     print(inputs_token)

with torch.no_grad():
    predict_context = model.generate(inputs_token)
    print(tokenizer.decode((predict_context[0].tolist())))
