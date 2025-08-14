import os

from omegaconf import OmegaConf
from nextstep.models.nextstep.tokenization_nextstep import special_tokens_dict
from .data.pretrain_data_512px import data_info_list, val_data_info_list
from .nextstep_qwen14b_256px import config
from nextstep.model_zoos import MODEL_ZOOS

NEXTSTEP_MODEL_PATH = MODEL_ZOOS["stepfun-ai/NextStep-1.1-Pretrain-256px"]
LM_MODEL_PATH = None
VAE_MODEL_PATH = MODEL_ZOOS["stepfun-ai/NextStep-1-f8ch16-Tokenizer"]
TOKENIZER_MODEL_PATH = None
OUTPUT_DIR = "./outputs/nextstep_qwen14b_512px"
RESUME_DIR = OUTPUT_DIR

config.model.model_name_or_path = NEXTSTEP_MODEL_PATH
config.model.lm_model_name_or_path = LM_MODEL_PATH
config.model.vae_name_or_path = VAE_MODEL_PATH
config.model.tokenizer_name_or_path = TOKENIZER_MODEL_PATH

config.data.image_size = 512
# Reduce num_workers and prefetch_factor to avoid DataLoader worker crashes
# If issues persist, further reduce to num_workers=1 or num_workers=0
config.data.pin_memory = True
config.data.num_workers = 2  # Reduced from default 4 to 2 to reduce resource contention
config.data.prefetch_factor = 4  # Reduced from default 8 to 4 to reduce memory usage

config.training.learning_rate = 1e-5
config.training.per_device_train_batch_size = 1
config.training.output_dir = OUTPUT_DIR
config.training.resume = RESUME_DIR
config.training.run_name = "nextstep_qwen14b_512px"
config.training.max_steps = 10_000
config.training.warmup_steps = 1000
config.training.save_steps = 1000
config.training.eval_steps = 1000

# Two ways to launch training:
# sudo smartrun -m configs.nextstep_qwen14b_512px
# sudo torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 -m configs.nextstep_qwen14b_512px
if __name__ == "__main__":
    from nextstep.engine.train_nextstep_ds import Arguments, main
    from nextstep.lazy_config import LazyLaunch
    from nextstep.lazyrun import setup_config, setup_environ

    # Setup smartrun environment variables (takes effect even when launched with torchrun)
    setup_config()
    setup_environ()

    LazyLaunch(main, Arguments)