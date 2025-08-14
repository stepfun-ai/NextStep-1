# VAE Tokenizer for image tokenization
huggingface-cli download --resume-download --local-dir-use-symlinks False stepfun-ai/NextStep-1-f8ch16-Tokenizer --resume-download --local-dir ./nextstep_models/NextStep-1-f8ch16-Tokenizer

# Qwen2.5-3B (optional) for fast pretraining
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2.5-3B --resume-download --local-dir ./nextstep_models/Qwen2.5-3B

# Qwen2.5-14B for pretraining
huggingface-cli download --resume-download --local-dir-use-symlinks False Qwen/Qwen2.5-14B --resume-download --local-dir ./nextstep_models/Qwen2.5-14B

# NextStep-1.1-Pretrain-256px is a text-to-image pretrain model with 256px images. It starts from the Qwen2.5-14B model, and is trained on 256px images with 500K steps with learning rate 1e-4
huggingface-cli download --resume-download --local-dir-use-symlinks False stepfun-ai/NextStep-1.1-Pretrain-256px --resume-download --local-dir ./nextstep_models/NextStep-1.1-Pretrain-256px


# NextStep-1.1-Pretrain is a text-to-image pretrain model with 512px images. It starts from the NextStep-1.1-Pretrain-256px model, and is trained on 512px images with 20K + 20K(Annealing) steps with learning rate 1e-5
huggingface-cli download --resume-download --local-dir-use-symlinks False stepfun-ai/NextStep-1.1-Pretrain --resume-download --local-dir ./nextstep_models/NextStep-1.1-Pretrain


# NextStep-1.1 is a text-to-image post-training model. It starts from the NextStep-1.1-Pretrain model, and is trained on 512px images with 1K steps with learning rate 5e-6 with NextStep-Grpo
huggingface-cli download --resume-download --local-dir-use-symlinks False stepfun-ai/NextStep-1.1 --resume-download --local-dir ./nextstep_models/NextStep-1.1


# The following models are from the first version (NextStep-1). Their performance is not as good as NextStep-1.1, so we do not recommend using them.

# NextStep-1-Large-Pretrain is a text-to-image pretrain model. It starts from the Qwen2.5-14B model, and is trained on 256px images with 300K steps (learning rate 1e-4) and 512px images with 100K steps (learning rate 1e-5)
# huggingface-cli download --resume-download --local-dir-use-symlinks False stepfun-ai/NextStep-1-Large-Pretrain --resume-download --local-dir ./nextstep_models/NextStep-1-Large-Pretrain


# NextStep-1-Large is a text-to-image model. It starts from the NextStep-1-Large-Pretrain model, and is trained on 512px images with 20K steps(Annealing) with learning rate 1e-5
# huggingface-cli download --resume-download --local-dir-use-symlinks False stepfun-ai/NextStep-1-Large --resume-download --local-dir ./nextstep_models/NextStep-1-Large

# NextStep-1-Large-Edit is a image editing model. It starts from the NextStep-1-Large model, and is trained on 512px images with 20K steps with learning rate 1e-5
# huggingface-cli download --resume-download --local-dir-use-symlinks False stepfun-ai/NextStep-1-Large-Edit --resume-download --local-dir ./nextstep_models/NextStep-1-Large-Edit
