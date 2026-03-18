# Yelp Reviews AI Assignment

This project explores how an instruction-tuned LLM performs on Yelp review understanding beyond simple classification. Instead of just predicting sentiment, I tried to evaluate how prompting strategies, reasoning styles, and robustness affect the model’s behavior in more realistic scenarios.

---

## What I Did

I broke the assignment into a few clear parts:

- Started with **1–5 star classification** using zero-shot and few-shot prompting  
- Compared **direct prompting vs chain-of-thought (CoT)** to see if reasoning actually helps  
- Extended the system into a **multi-objective assistant** that:
  - predicts star rating  
  - extracts the main complaint/compliment  
  - generates a short business response  
- Tested **domain shift** using Amazon reviews  
- Evaluated **robustness** by adding noise like typos, repetition, punctuation changes, and prompt-injection-style text  

---

## Setup

- Model: `llama4:latest` (served via Ollama-compatible endpoint)  
- Temperature: 0.0 (to keep outputs consistent)  
- Outputs: Strict JSON format with parsing + validation  

---

## Key Observations

- **Few-shot prompting helped slightly**, but the gain was small  
- **Direct prompting worked better than CoT** in this setup — the model sometimes “overthought” and made worse predictions with reasoning  
- The assistant extension worked reasonably well for clear reviews, but struggled with mixed or ambiguous ones  
- **Domain shift was noticeable** (performance dropped when moving from Yelp to Amazon)  
- The model handled simple noise well, but **prompt injection had the biggest impact on predictions**  

---

## Project Structure

```
├── notebooks/
│   ├── 01_dataset_loading_and_preparation.ipynb
│   ├── 02_zero_shot_vs_few_shot.ipynb
│   ├── 03_direct_vs_cot.ipynb
│   ├── 04_multi_objective_assistant.ipynb
│   └── 05_domain_shift_and_robustness.ipynb
│
├── src/
│   ├── prompts.py
│   ├── llm_runner.py
│   ├── evaluation.py
│   └── config.py
│
├── data/
│   ├── processed/
│
├── outputs/
├── reports/
```

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

1. Start Jupyter:

```bash
jupyter notebook
```

1. Run notebooks in order:

- 01 → dataset prep  
- 02 → zero vs few-shot  
- 03 → direct vs CoT  
- 04 → assistant  
- 05 → domain shift & robustness  

---

## Notes

- I used subsets of the dataset for faster iteration instead of full 650k training data  
- Amazon polarity dataset was mapped to {1, 5} for rough comparison, so results should be interpreted carefully  
- Robustness tests are synthetic and don’t fully represent real-world noise  

---

## Final Thought

The most interesting takeaway for me was that more complex prompting (like CoT) didn’t always help. In this case, simpler and more direct prompts actually gave better results. At the same time, the model was quite sensitive to domain changes and prompt injection, which shows there’s still a gap between controlled experiments and real-world usage.
