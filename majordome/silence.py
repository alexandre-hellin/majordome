import os
import warnings

# Disable loading bars
os.environ["TQDM_DISABLE"] = "1"

# Set Transformers verbosity to error
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Silence some Python Warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*sequentially on GPU.*")
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")

# After import
import transformers
transformers.logging.set_verbosity_error()

# Disable loading bars for HF
from huggingface_hub import utils as hf_utils
hf_utils.disable_progress_bars()