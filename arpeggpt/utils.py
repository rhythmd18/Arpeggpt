import torch

def midi_to_token_ids(midi_file, tokenizer):
    encoded = tokenizer(midi_file)
    encoded_tensor = torch.tensor(encoded)
    return encoded_tensor

def token_ids_to_midi(token_ids, tokenizer):
    generated_midi = tokenizer(token_ids)
    return generated_midi