# import os
# from dotenv import load_dotenv
from pathlib import Path
from miditok import REMI, TokenizerConfig, TokSequence
import shutil
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from arpeggpt.config import get_config

# load_dotenv()
# hf_token = os.getenv('HF_TOKEN')

# tokenizer = REMI.from_pretrained('algorhythmic/remi-giantmidi-tokenizer', token=hf_token)
# tokenizer.save_pretrained('arpeggpt/midi_tokenizer')
tokenizer = REMI.from_pretrained('arpeggpt/midi_tokenizer')
midi_dir = Path('data/giantmidi-small').resolve()
midi_paths = list(midi_dir.glob("**/*.mid"))
dataset_chunks_dir = Path('data/chunks').resolve()

def prepare_dataset(batch_size, config, rebuild=False):
    if rebuild and dataset_chunks_dir.exists():
        shutil.rmtree(dataset_chunks_dir)
    if (not dataset_chunks_dir.exists()
        or not any(dataset_chunks_dir.rglob('*.mid'))):
        dataset_chunks_dir.mkdir(parents=True, exist_ok=True)
        split_files_for_training(
            files_paths=midi_paths,
            tokenizer=tokenizer,
            save_dir=dataset_chunks_dir,
            max_seq_len=config['context_length']
        )
    dataset = DatasetMIDI(
        files_paths=list(dataset_chunks_dir.glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=config['context_length'],
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"],
    )
    collator = DataCollator(
        tokenizer.pad_token_id, 
        copy_inputs_as_labels=True,
        shift_labels=True
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collator,
        shuffle=False
    )
    return dataloader