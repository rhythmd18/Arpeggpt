# Arpeggpt

Arpeggpt is a transformer model built from scratch that generates classical piano compositions. It has been pre-trained on the [GiantMidi-Piano](https://github.com/bytedance/GiantMIDI-Piano) dataset, which is a piano dataset consisting of 10,855 MIDI files of 2,786 composers. 
> This repository is currently at a configurable state where users can bring in their own piano dataset in the MIDI format and can train the model based on multiple GPT configurations. This README explains how to set up the training process and invoke the CLI to train models of different predefined sizes: `test`, `small`, `medium`, `large`, and `xl`.

---

## 1. Quick Start

```bash
# Clone and enter the repo
git clone https://github.com/rhythmd18/Arpeggpt.git
cd Arpeggpt

# (Optional) create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Data Preparation

- Create a `data/` directory.

```bash
mkdir data
cd data
```

- Place your training data in there. For example, in my case:

```
data/
  giantmidi/
  giantmidi-small/
  ...
```

---

## 3. Configuration Presets

Each model size corresponds to a preset configuration controlling:

| Size   | Layers (`n_layers`) | Embedding Dims (`emb_dim`) | Heads (`n_heads`) |
|--------|---------------------|----------------------------|-------------------|
| test   | 12                  | 192                        | 12                |
| small  | 12                  | 768                        | 12                |
| medium | 24                  | 1024                       | 16                |
| large  | 36                  | 1260                       | 20                |
| xl     | 48                  | 1600                       | 25                |

These configurations have been defined in `arpeggpt/config.py`

---

## 4. Training Commands

### 4.1 Train with the default configuration (Only for quick testing and debugging)

```bash
python main.py
```

Flags that define training configurations:

| Flag | Meaning | Default |
|------|---------|---------|
| `--config <str>` | Define which GPT configuration to use (one of `test`, `small`, `medium`, `large`, and `xl`) | `test` |
| `--batch-size <int>` | The batch size | 16 |
| `--num-epochs <epochs>` | The number of epochs to train for | 20 |
| `--save-every <n>` | Save the model every n epochs | 4 |

### 4.2 Small

```bash
python main.py --config small
```

### 4.3 Medium

```bash
python main.py --config medium
```

### 4.4 Large

```bash
python main.py --config large
```

### 4.5 XL

```bash
python main.py --config xl
```

> Adjust batch size, number of epochs, and checkpointing frequency using the flags defined above. For example:
```bash
python main.py --config medium --batch-size 128 --num-epochs 100 --save-every 5
```

---

## 5. Attribution

[LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
