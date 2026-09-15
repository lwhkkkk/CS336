import torch
from cs336_basics.tokenizer import Tokenizer 
from cs336_basics.model import TransformerLM ,sample_decoding 

checkpoints = "checkpoints/checkpoint_20000.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = Tokenizer.from_files("cs336_basics/tinystories_vocab.pkl","cs336_basics/tinystories_merges.pkl",special_tokens =["<|endoftext|>"])

model = TransformerLM(
    vocab_size = 10001,
    context_length =256,
    d_model = 512,
    num_layers = 4,
    num_heads =16,
    d_ff = 1344,
    rope_theta = 10000.0,
    device = device 
)

checkpoint_dict  =torch.load(checkpoints, map_location=device)
model.load_state_dict(checkpoint_dict["model"])
model.eval()

prompt_text = "Once upon a time"
prompt_ids = tokenizer.encode(prompt_text)


#模型期待接受张量形状是(batch_size, sequence_length)
prompt_tensor = torch.tensor([prompt_ids], device =device, dtype =torch.long)

#获取终止符ID
eos_id = tokenizer.inverse_vocab[b"<|endoftext|>"]

output = sample_decoding(
    model = model,
    max_new_tokens = 256,
    eos_token_id = eos_id,
    prompt_indices = prompt_tensor,
    temperature = 0.8,
    top_p = 0.9
)

#拉成list
output_ids = output[0].tolist()
generated_text = tokenizer.decode(output_ids)
print(generated_text)