import torch
from pathlib import Path
from arpeggpt.model import GPTModel
from arpeggpt.config import get_config
from utils import midi_to_token_ids, token_ids_to_midi
from miditok import REMI


def generate_midi(
    model, idx, max_new_tokens, context_size,
    temperature=0.0, top_k=None, eos_id=None, attn_mask=None
):
    # idx is (batch, n_tokens) array of indices in the current context
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond, attn_mask)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdims=True)

        if idx_next == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


if __name__ == '__main__':
    tokenizer = REMI.from_pretrained('arpeggpt/midi_tokenizer')
    config = get_config('small')

    sample_midi_path = list(Path('/kaggle/input/samplemidi').glob("**/*.mid"))
    start_midi = sample_midi_path[0]

    model = GPTModel()
    model.eval()

    token_ids = generate_midi(
        model=model,
        idx=midi_to_token_ids(start_midi, tokenizer),
        max_new_tokens=500,
        context_size=config['context_length'],
        temperature=1.4,
        top_k=25
    )
    generated_midi = token_ids_to_midi(token_ids, tokenizer)
    generated_midi.dump(Path('Sample.mid'))