from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = BASE_DIR / "outputs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

RANDOM_SEED = 42

# Ollama Config
OLLAMA_BASE_URL = "https://ollama.merai.app"
DEFAULT_MODEL = "llama4:latest"

TRAIN_FULL_FILE = PROCESSED_DATA_DIR / "yelp_train_full.csv"
TEST_FULL_FILE = PROCESSED_DATA_DIR / "yelp_test_full.csv"
PROMPT_SUBSET_FILE = PROCESSED_DATA_DIR / "yelp_prompt_subset_500.csv"
ASSISTANT_SUBSET_FILE = PROCESSED_DATA_DIR / "yelp_assistant_subset_100.csv"
ROBUSTNESS_SUBSET_FILE = PROCESSED_DATA_DIR / "yelp_robustness_subset_200.csv"
SUMMARY_FILE = REPORTS_DIR / "dataset_summary.txt"

PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

ZERO_SHOT_PRED_FILE = PREDICTIONS_DIR / "zero_shot_predictions.csv"
FEW_SHOT_PRED_FILE = PREDICTIONS_DIR / "few_shot_predictions.csv"
COT_PRED_FILE = PREDICTIONS_DIR / "cot_predictions.csv"
ASSISTANT_PRED_FILE = PREDICTIONS_DIR / "assistant_predictions.csv"
ASSISTANT_EVAL_FILE = PREDICTIONS_DIR / "assistant_evaluation_sample.csv"
DOMAIN_SHIFT_FILE = PREDICTIONS_DIR / "domain_shift_predictions.csv"
ROBUSTNESS_FILE = PREDICTIONS_DIR / "robustness_predictions.csv"

