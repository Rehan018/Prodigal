from textwrap import dedent
def build_zero_shot_prompt(review_text: str) -> str:
    return dedent(f"""
    You are a strict sentiment classifier for Yelp reviews.

    Task:
    Classify the review into an integer star rating from 1 to 5.

    Output rules:
    - Return only valid JSON
    - Do not include markdown
    - Do not include extra text
    - Use exactly these keys: "stars", "explanation"
    - "stars" must be an integer from 1 to 5
    - "explanation" must be short, specific, and grounded in the review

    Required format:
    {{"stars": 3, "explanation": "Mixed sentiment with mild praise and some dissatisfaction."}}

    Review:
    {review_text}
    """).strip()


def build_assistant_prompt(review_text: str) -> str:
    return f"""
You are an AI assistant helping businesses understand Yelp reviews.

Your task is to read the review and return:
1. A star rating from 1 to 5
2. The main complaint or compliment as a short key point
3. A short, polite business response

Return only valid JSON in exactly this format:
{{
  "stars": 3,
  "key_point": "Main complaint or compliment from the review",
  "business_response": "Short polite response grounded in the review"
}}

Rules:
- stars must be an integer from 1 to 5
- key_point must be concise, specific, and based only on the review
- business_response must be short, polite, and relevant
- do not invent facts not present in the review
- do not include markdown
- do not include extra text outside JSON

Review:
{review_text}
""".strip()


def build_direct_prompt(review_text: str) -> str:
    return f"""
You are a strict sentiment classifier for Yelp reviews.

Classify the review into 1 to 5 stars.

Return only valid JSON:
{{"stars": 3, "explanation": "Short explanation"}}

Review:
{review_text}
""".strip()

def build_cot_prompt(review_text: str) -> str:
    return f"""
You are a strict sentiment classifier for Yelp reviews.

First reason briefly about the sentiment, then provide the final star rating.

Return only valid JSON with exactly these keys:
{{"reasoning": "brief reasoning", "stars": 3, "explanation": "short explanation"}}

Rules:
- stars must be an integer from 1 to 5
- reasoning must be grounded in the review
- explanation must be short
- return JSON only

Review:
{review_text}
""".strip()

def build_few_shot_prompt(review_text: str) -> str:
    return dedent(f"""
    You are a strict sentiment classifier for Yelp reviews.

    Task:
    Classify the review into an integer star rating from 1 to 5.

    Output rules:
    - Return only valid JSON
    - Do not include markdown
    - Do not include extra text
    - Use exactly these keys: "stars", "explanation"
    - "stars" must be an integer from 1 to 5
    - "explanation" must be short, specific, and grounded in the review

    Examples:

    Review: The food was cold, the waiter was rude, and the place was dirty. I will not come back.
    Output: {{"stars": 1, "explanation": "Strongly negative review mentioning bad food, rude service, and poor cleanliness."}}

    Review: The place was okay. Service was acceptable and the food was decent, but nothing stood out.
    Output: {{"stars": 3, "explanation": "Neutral review with mild satisfaction but no strong praise or complaint."}}

    Review: Amazing food, friendly staff, and quick service. One of the best places I have visited.
    Output: {{"stars": 5, "explanation": "Strongly positive review with enthusiastic praise for food, staff, and service."}}

    Now classify this review.

    Required format:
    {{"stars": 3, "explanation": "Mixed sentiment with mild praise and some dissatisfaction."}}

    Review:
    {review_text}
    """).strip()