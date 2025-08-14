import os

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = "path/to/your/results_folder"

# define the benchmarks to run
BENCHMARKS = {
    "GenEval": {
        "inference_manager": "GenEvalInferenceManager",
        "bench_name": ["GENEVAL"],
    },
    "DPG": {
        "inference_manager": "DPGBenchInferenceManager",
        "bench_name": ["DPG_BENCH"],
    },
    "GenAI": {
        "inference_manager": "GENAIBenchInferenceManager",
        "bench_name": ["GENAI_BENCH"],
    },
    "WISE": {
        "inference_manager": "WISEInferenceManager",
        "bench_name": ["WISE_CULTURAL_COMMON_SENSE", "WISE_NATURAL_SCIENCE", "WISE_SPATIO_TEMPORAL_REASONING"],
    },
    "OneIG": {
        "inference_manager": "ONEIGBenchInferenceManager",
        "bench_name": ["ONEIG_BENCH_EN", "ONEIG_BENCH_ZH"],
    },
}

T2I_BENCHMARK = {
    # GenEval
    "GENEVAL": os.path.join(_ROOT_DIR, "prompts/GenEval.jsonl"),
    # DPG Bench
    "DPG_BENCH": os.path.join(_ROOT_DIR, "prompts/dpg_prompts"),
    # GenAI Bench
    "GENAI_BENCH": os.path.join(_ROOT_DIR, "prompts/genai_image.json"),
    # WISE
    "WISE_CULTURAL_COMMON_SENSE": os.path.join(_ROOT_DIR, "prompts/WISE_cultural_common_sense.json"),
    "WISE_NATURAL_SCIENCE": os.path.join(_ROOT_DIR, "prompts/WISE_natural_science.json"),
    "WISE_SPATIO_TEMPORAL_REASONING": os.path.join(_ROOT_DIR, "prompts/WISE_spatio_temporal_reasoning.json"),
    # ONEIG Bench
    "ONEIG_BENCH_EN": os.path.join(_ROOT_DIR, "prompts/OneIG-Bench.csv"),
    "ONEIG_BENCH_ZH": os.path.join(_ROOT_DIR, "prompts/OneIG-Bench-ZH.csv"),
}