import os


class IdentityDict(dict):
    def __missing__(self, key):
        if key is None:
            return None
        return key

MODEL_ZOOS = IdentityDict(
    {
        # VAE tokenizer model for tokenizing images
        "stepfun-ai/NextStep-1-f8ch16-Tokenizer" : "./nextstep_models/NextStep-1-f8ch16-Tokenizer",
        # Qwen2.5-3B model for simple pretraining
        "Qwen/Qwen2.5-3B"                                : "./nextstep_models/Qwen2.5-3B",
        # Qwen2.5-14B model for standard pretraining
        "Qwen/Qwen2.5-14B"                         : "./nextstep_models/Qwen2.5-14B",
        # pretrain model with 256px images with 500K steps with learning rate 1e-4
        "stepfun-ai/NextStep-1.1-Pretrain-256px" : "./nextstep_models/NextStep-1.1-Pretrain-256px",
        # pretrain model with 500K steps 256px, 20K steps 512px, and 20K steps Annealing
        "stepfun-ai/NextStep-1.1-Pretrain" : "./nextstep_models/NextStep-1.1-Pretrain",  
        # NextStep-1.1 model for inference
        "stepfun-ai/NextStep-1.1" : "./nextstep_models/NextStep-1.1",
        # NextStep-1-Large-Pretrain (v1.0, not recommended)
        "stepfun-ai/NextStep-1-Large-Pretrain" : "./nextstep_models/NextStep-1-Large-Pretrain",
        # NextStep-1-Large (v1.0, not recommended)
        "stepfun-ai/NextStep-1-Large" : "./nextstep_models/NextStep-1-Large",
        # NextStep-1-Large-Edit (v1.0, not recommended)
        "stepfun-ai/NextStep-1-Large-Edit" : "./nextstep_models/NextStep-1-Large-Edit",
    }
)

PATH_TO_MODEL_NAME = IdentityDict({v: k for k, v in MODEL_ZOOS.items()})
