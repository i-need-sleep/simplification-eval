from transformers import AutoTokenizer, AutoModel
import globals as uglobals

pretrained_dir = f'../{uglobals.PRETRAINED_DIR}'

tokenizer_name = 'microsoft/deberta-v2-xlarge'
save_name = 'deberta'
AutoTokenizer.from_pretrained(tokenizer_name, force_download=True).save_pretrained(f'{pretrained_dir}/tokenizers/{save_name}')
AutoModel.from_pretrained(tokenizer_name, force_download=True).save_pretrained(f'{pretrained_dir}/checkpoints/{save_name}')