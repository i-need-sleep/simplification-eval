from transformers import AutoTokenizer, AutoModel
import globals as uglobals

pretrained_dir = f'../{uglobals.PRETRAINED_DIR}'

model_name = 'roberta-large'
save_name = 'roberta'
AutoTokenizer.from_pretrained(model_name, force_download=True).save_pretrained(f'{pretrained_dir}/tokenizers/{save_name}')
AutoModel.from_pretrained(model_name, force_download=True).save_pretrained(f'{pretrained_dir}/checkpoints/{save_name}')