# NextStep Package

The `nextstep/` package contains the core training code, where the entire training pipeline is implemented.

---

## Submodules

- `lazyrun.py`: Launch wrapper for environment variables, distributed parameters, and debug mode
- `lazy_config/`: Configuration parsing, command-line overrides, and run directory initialization
- `engine/`: Training and validation main loop
- `data/`: Tar indexing, decoding, and warmup utilities
- `datasets/`: Multiple dataset adapters and mixed sampling
- `models/`: NextStep model implementation and special tokens
- `service/`: Streamlit-based data preview service
- `utils/`: Logging, communication, schedulers, optimizers, and monitoring utilities

---

## Key Entry Points (Execution Order)

1. `nextstep/lazyrun.py`
2. `nextstep/lazy_config/arg_parser.py`
3. `nextstep/engine/train_nextstep_ds.py`
4. `nextstep/datasets/mixed_dataset.py`
5. `nextstep/models/nextstep/modeling_nextstep.py`
