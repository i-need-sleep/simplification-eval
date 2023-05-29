# Data
DATA_DIR = '../data'
RAW_DIR = f'{DATA_DIR}/raw'

OPENWEBTEXT_DIR = f'{RAW_DIR}/openwebtext'
OPENWEBTEXT_DIR_ALT = f'../../syntax_acquisition/data/openwebtext'
TURKCORPUS_DIR = f'{RAW_DIR}/turkcorpus'

# Stage 1
PROCESSED_DIR = f'{DATA_DIR}/processed'
SIM_OPENWEBTEXT_DIR = f'{PROCESSED_DIR}/openwebtext'

# Stage 2
STAGE2_DIR = f'{DATA_DIR}/stage2'
STAGE2_OUTPUTS_DIR = f'{STAGE2_DIR}/outputs'
STAGE2_RAW = f'{STAGE2_DIR}/raw'
STAGE_2_REFERENCED = f'{STAGE2_OUTPUTS_DIR}/referenced'
STAGE_2_SYSTEMS = f'{STAGE2_OUTPUTS_DIR}/systems'
STAGE_2_WIKI_LARGE_REF = f'{STAGE_2_SYSTEMS}/dress/WikiLarge/test/Reference'
STAGE_2_WIKI_LARGE_SRC = f'{STAGE_2_SYSTEMS}/dress/WikiLarge/test/Complex'
DRESS_DIR = f'{STAGE_2_SYSTEMS}/dress'

# Stage 3
STAGE3_DIR = f'{DATA_DIR}/stage3'
STAGE3_PROCESSED_DIR = f'{STAGE3_DIR}/processed'

# Results
RESULTS_DIR = '../results'
CHECKPOINTS_DIR = f'{RESULTS_DIR}/checkpoints'
OUTPUTS_DIR = f'{RESULTS_DIR}/outputs'

# Resources
PRETRAINED_DIR = '../pretrained'
DERBERTA_MODEL_DIR = f'{PRETRAINED_DIR}/checkpoints/deberta'
DERBERTA_TOKENIZER_DIR = f'{PRETRAINED_DIR}/tokenizers/deberta'
LENS_DIR = f'{PRETRAINED_DIR}/LENS/checkpoints/epoch=5-step=6102.ckpt'