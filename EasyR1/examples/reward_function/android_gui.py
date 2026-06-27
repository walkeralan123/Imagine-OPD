"""
Number Game Reward Function

Reward rules:
- Correct number selection: +1.0
- Incorrect number selection: 0.0

Input format:
reward_input = {
    "response": "1",  # Model answer (0/1/2)
    "response_length": 10,  # Response length in tokens
    "ground_truth": "1"  # Correct answer (0/1/2)
}

Output format:
{
    "overall": 1.0,  # Overall score (required)
    "accuracy": 1.0  # Accuracy metric (optional)
}
"""

import re
from typing import Any


# Metadata required by EasyR1
REWARD_NAME = "number_game"
REWARD_TYPE = "batch"  # Batch mode


def extract_answer(response: str) -> str:
    """
    Extract the answer index from the model response.

    Args:
        response: raw model response

    Returns:
        "0", "1", "2", or "" when extraction fails.
    """
    # Case 1: the response is a single digit
    response = response.strip()
    if response in ["0", "1", "2"]:
        return response

    # Case 2: extract the first 0/1/2 from extra text
    match = re.search(r"[012]", response)
    if match:
        return match.group(0)

    # Extraction failed
    return ""


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    """
    Compute scores for a batch of samples.

    Args:
        reward_inputs: list of samples, each containing:
            - response: model response
            - response_length: response length
            - ground_truth: correct answer

    Returns:
        List of score dictionaries, each containing:
            - overall: score, where 1.0 is correct and 0.0 is incorrect
            - accuracy: same as overall, used for monitoring
    """
    scores = []

    for reward_input in reward_inputs:
        response = reward_input.get("response", "")
        ground_truth = reward_input.get("ground_truth", "")

        # Extract answer
        predicted = extract_answer(response)

        # Compute score
        if predicted == ground_truth:
            score = 1.0
        else:
            score = 0.0

        # Return format must include overall
        scores.append({"overall": score, "accuracy": score})

    return scores


# Test cases
if __name__ == "__main__":
    test_cases = [
        # Exact match
        {"response": "0", "response_length": 1, "ground_truth": "0"},
        {"response": "1", "response_length": 1, "ground_truth": "1"},
        {"response": "2", "response_length": 1, "ground_truth": "2"},
        # Response with extra text
        {"response": "The answer is 1", "response_length": 15, "ground_truth": "1"},
        {"response": "I choose option 2", "response_length": 18, "ground_truth": "2"},
        # Wrong answer
        {"response": "0", "response_length": 1, "ground_truth": "1"},
        {"response": "2", "response_length": 1, "ground_truth": "0"},
        # Extraction failed
        {"response": "I don't know", "response_length": 12, "ground_truth": "1"},
        {"response": "", "response_length": 0, "ground_truth": "2"},
    ]

    scores = compute_score(test_cases)

    print("Reward Function Test Results:")
    print("=" * 60)
    for i, (test, score) in enumerate(zip(test_cases, scores), 1):
        print(f"{i}. Response: {test['response']!r}")
        print(f"   Ground Truth: {test['ground_truth']!r}")
        print(f"   Score: {score}")
        print()
