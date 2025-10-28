# -*- coding: utf-8 -*-
"""
游戏领域评估器
"""

import json
from typing import List
from openai import OpenAI


class GameCorrectnessEvaluator:
    """游戏领域正确性评估器"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o", temperature: float = 0.01):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature

        self.client = OpenAI(api_key=api_key, base_url=base_url)

        # 正确性评估prompt
        self.evaluation_prompt = """You are an extremely strict game knowledge correctness evaluator. Your task is to evaluate the accuracy of a predicted answer against the ground truth answer for a game-related question with MAXIMUM RIGOR.

ULTRA-STRICT EVALUATION CRITERIA:

1. FACTUAL ACCURACY (ZERO TOLERANCE):
   - Every single fact about game mechanics, rules, systems, and lore must be 100% correct
   - ANY factual error, no matter how minor, significantly impacts the score
   - Factual claims not supported by the ground truth are considered errors

2. NUMERICAL INFORMATION (EXACT PRECISION):
   - All numbers, statistics, values, quantities, percentages must be exactly correct
   - Even tiny numerical discrepancies (>±2%) are heavily penalized
   - Approximations are only acceptable if explicitly indicated as such

3. TERMINOLOGY AND NAMES (PERFECT ACCURACY):
   - Character names, location names, item names, ability names, and ALL game-specific terms must be spelled exactly correctly
   - Any misspelling or incorrect terminology significantly reduces the score
   - Synonyms are only acceptable if they are commonly recognized alternatives

4. COMPLETENESS AND COVERAGE (COMPREHENSIVE):
   - The answer must address EVERY aspect of the question thoroughly
   - Missing ANY critical information mentioned in the ground truth is a major defect
   - Partial answers that leave important aspects unanswered are heavily penalized

5. ADDITIONAL INFORMATION (STRICT VERIFICATION):
   - Any extra information beyond the ground truth must be 100% accurate and verifiable
   - Speculative content, unsupported claims, or hallucinations result in immediate score reduction
   - Information that contradicts or confuses the ground truth is unacceptable

ULTRA-STRICT 3-LEVEL SCORING SYSTEM:

- 2 (Exceptionally Perfect):
  * EVERY fact is 100% accurate with zero errors
  * ALL numbers and terminology are precisely correct
  * Comprehensively addresses the question with complete coverage
  * Any additional information is verified accurate and genuinely helpful
  * No ambiguity, no errors, no omissions - truly exemplary answer

- 1 (Acceptable with Minor Flaws):
  * Core facts are accurate but contains 1-2 very minor issues
  * Slight numerical discrepancies (≤±2%) OR missing 1-2 non-essential details
  * Addresses the main question adequately but may lack some depth
  * Minor terminology issues that don't materially affect understanding
  * Overall sound but not perfect

- 0 (Defective/Inadequate):
  * Contains ANY significant factual errors or multiple minor errors
  * Notable numerical inaccuracies (>±2%) or missing important quantitative information
  * Fails to address key aspects of the question or provides incomplete coverage
  * Contains questionable information, contradictions, or unsupported claims
  * Any hallucination, fabrication, or misleading content

Question: {question}
Ground Truth Answer: {answers}
Predicted Answer: {prediction}

Apply MAXIMUM STRICTNESS in your evaluation. Score 2 should be EXTREMELY RARE and reserved only for truly flawless answers. Score 1 should be given only to genuinely good answers with minimal, non-critical flaws. Score 0 for everything else, including answers that are "mostly correct" but have clear defects. Remember: being strict protects the integrity of the evaluation. Return your evaluation as a JSON object with the "accuracy" field (0, 1, or 2)."""

        # Faithfulness评估prompt
        self.faithfulness_prompt = """You are an extremely strict faithfulness evaluator for game-related question answering systems. Your task is to evaluate whether the predicted answer is ENTIRELY FAITHFUL to the provided retrieved context documents with MAXIMUM RIGOR.

ULTRA-STRICT FAITHFULNESS EVALUATION CRITERIA:

1. INFORMATION SOURCE VERIFICATION (ZERO TOLERANCE):
   - EVERY piece of information in the predicted answer MUST be directly supported by the retrieved context
   - ANY claim, fact, or detail not found in the provided contexts is considered a faithfulness violation
   - Inferences or logical deductions that go beyond what's explicitly stated in contexts are NOT allowed

2. FACTUAL CONSISTENCY (EXACT ALIGNMENT):
   - All facts, numbers, names, dates, and details must match EXACTLY with the context
   - No paraphrasing that changes meaning or introduces ambiguity
   - No combining information from different contexts in ways that create new unsupported claims

3. CONTEXT GROUNDING (MANDATORY SUPPORT):
   - Each statement in the answer must be traceable to specific parts of the retrieved contexts
   - Information synthesis is only acceptable if it directly reflects what's stated in the contexts
   - No external knowledge beyond what's provided in the contexts

4. HALLUCINATION DETECTION (ZERO TOLERANCE):
   - Any information not present in the contexts is considered hallucination
   - This includes reasonable-sounding but unverified details about game mechanics, characters, locations, etc.
   - Even "common knowledge" about games must be present in contexts to be considered faithful

5. OMISSION vs ADDITION PRINCIPLE:
   - It's better to omit information not in contexts than to add unsupported information
   - Incomplete but faithful answers are preferred over complete but unfaithful ones

ULTRA-STRICT 3-LEVEL SCORING SYSTEM:

- 2 (Perfectly Faithful):
  * EVERY statement in the answer is directly supported by the retrieved contexts
  * No hallucinations, no unsupported claims, no external information
  * Perfect alignment between answer content and context information
  * May be incomplete but everything stated is verifiable from contexts

- 1 (Mostly Faithful with Minor Issues):
  * Core information is supported by contexts but contains 1-2 minor unsupported details
  * Slight paraphrasing that doesn't change meaning significantly
  * Minor inferences that are very close to what's stated in contexts
  * Overall faithful but not perfectly grounded

- 0 (Unfaithful/Hallucinated):
  * Contains significant information not found in the retrieved contexts
  * Multiple unsupported claims or facts
  * Introduces external knowledge not present in contexts
  * Creates new information through inappropriate synthesis
  * Any clear hallucination or fabrication

Question: {question}
Retrieved Contexts: {contexts}
Predicted Answer: {prediction}

Apply MAXIMUM STRICTNESS in evaluating faithfulness. Score 2 should be EXTREMELY RARE and only for answers that are completely verifiable from the contexts. Score 1 only for answers that are largely faithful with minimal unsupported content. Score 0 for everything else that contains unverified information. Remember: faithfulness means the answer can ONLY contain information that is explicitly or very clearly implied in the provided contexts. Return your evaluation as a JSON object with the "faithfulness" field (0, 1, or 2)."""

    def evaluate_single(self, question: str, ground_truth: str, prediction: str) -> float:
        """评估单个问答对的正确性"""
        full_prompt = self.evaluation_prompt.format(
            question=question,
            answers=ground_truth,
            prediction=prediction
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional game knowledge correctness evaluator. Always respond with valid JSON format."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content.strip()
        result_json = json.loads(result_text)
        accuracy = result_json.get('accuracy', 0)

        return float(accuracy) if accuracy in [0, 1, 2] else 0.0

    def evaluate_faithfulness(self, question: str, contexts: List[str], prediction: str) -> float:
        """评估单个问答对的faithfulness"""
        contexts_str = "\n\n".join([f"Context {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])
        full_prompt = self.faithfulness_prompt.format(
            question=question,
            contexts=contexts_str,
            prediction=prediction
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a professional faithfulness evaluator for game knowledge QA systems. Always respond with valid JSON format."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content.strip()
        result_json = json.loads(result_text)
        faithfulness = result_json.get('faithfulness', 0)

        return float(faithfulness) if faithfulness in [0, 1, 2] else 0.0
