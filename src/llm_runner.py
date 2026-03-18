import json
import re
import requests
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, Optional, Callable
from src.config import OLLAMA_BASE_URL, DEFAULT_MODEL


OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.0,
    num_predict: int = 250,
    timeout: int = 180,
) -> str:
    """
    Call the remote Ollama instance.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
        },
    }

    try:
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return ""


def extract_json_block(text: str) -> Optional[str]:
    """
    Try to extract the first JSON object from model output.
    """
    if not text:
        return None

    text = text.strip()

    # Direct JSON check
    if text.startswith("{") and text.endswith("}"):
        return text

    # Find first {...} using regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0)

    return None


def parse_prediction(raw_output: str) -> Dict[str, Any]:
    """
    Parse model output into structured prediction.
    """
    parsed = {
        "raw_output": raw_output,
        "json_valid": False,
        "stars_pred": None,
        "explanation_pred": None,
        "parse_error": None,
    }

    try:
        json_text = extract_json_block(raw_output)
        if json_text is None:
            parsed["parse_error"] = "No JSON object found"
            return parsed

        obj = json.loads(json_text)

        stars = obj.get("stars", None)
        explanation = obj.get("explanation", None)

        if not isinstance(stars, (int, float)):
            parsed["parse_error"] = "stars is not a number"
            return parsed

        stars = int(stars)
        if stars < 1 or stars > 5:
            parsed["parse_error"] = "stars out of range"
            return parsed

        parsed["json_valid"] = True
        parsed["stars_pred"] = stars
        parsed["explanation_pred"] = str(explanation).strip() if explanation else None
        return parsed

    except Exception as e:
        parsed["parse_error"] = str(e)
        return parsed


def evaluate_subset(
    df: pd.DataFrame, 
    prompt_builder: Callable[[str], str],
    model: str = DEFAULT_MODEL,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Iterate over dataframe rows, call LLM, and store results.
    """
    working_df = df.copy()
    if limit:
        working_df = working_df.head(limit)

    results = []
    
    # Using tqdm for progress tracking
    for _, row in tqdm(working_df.iterrows(), total=len(working_df), desc=f"Evaluating {model}"):
        prompt = prompt_builder(row["text"])
        raw_res = call_ollama(prompt, model=model)
        parsed = parse_prediction(raw_res)
        
        # Combine original data with prediction
        result_row = {
            "text": row["text"],
            "stars_true": row["stars"],
            "stars_pred": parsed["stars_pred"],
            "json_valid": parsed["json_valid"],
            "explanation": parsed["explanation_pred"],
            "error": parsed["parse_error"]
        }
        results.append(result_row)
        
    return pd.DataFrame(results)