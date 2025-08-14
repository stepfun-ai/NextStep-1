import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from evals.config import BENCHMARKS, RESULTS_PATH
from evals.gen_images import __all__


def main(args):
    from evals.inference_main import InferenceManager

    results_path = os.path.join(RESULTS_PATH, args.model_alias)
    os.makedirs(results_path, exist_ok=True)

    # initialize the inference manager
    inference_manager = InferenceManager(
        model_name_or_path=args.model_name_or_path,
        output_dir=results_path,
    )

    # inference for all benchmarks
    for _, bench_info in BENCHMARKS.items():
        manager_class = bench_info["inference_manager"]
        for bench_name in bench_info["bench_name"]:
            InferenceManager.methods["run_t2i_benchmark"][manager_class](inference_manager, bench_name)

    # finish and cleanup
    inference_manager.finish_and_cleanup()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--model_alias", type=str, default="nextstep-1")
    parser.add_argument("--model_type", type=str, default="nextstep")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    main(args)