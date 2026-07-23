"""
Automated RAGAS Evaluator.

Uses the RAGAS framework to compute core RAG metrics:
- Faithfulness (Target > 0.85)
- Answer Relevancy (Target > 0.80)
- Context Precision (Target > 0.75)
- Context Recall (Target > 0.70)
"""

import logging
import json
from typing import Dict, List, Any
from datasets import Dataset

try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RagasEvaluator:
    """
    Evaluates RAG pipeline outputs against a golden dataset using RAGAS metrics.
    """

    def __init__(self, llm=None, embeddings=None):
        """
        Initialize the RAGAS evaluator.
        
        Args:
            llm: Optional Langchain-compatible LLM object for RAGAS to use. 
                 If None, RAGAS defaults to OpenAI (requires OPENAI_API_KEY).
            embeddings: Optional Langchain-compatible embeddings object.
        """
        if not RAGAS_AVAILABLE:
            raise ImportError("The 'ragas' package is not installed. Please run `pip install ragas`.")
            
        self.llm = llm
        self.embeddings = embeddings
        
        # We enforce targets per our architecture specs
        self.targets = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.80,
            "context_precision": 0.75,
            "context_recall": 0.70,
        }

    def evaluate_dataset(self, evaluation_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a list of pipeline execution results.
        
        Expected evaluation_data format:
        [
            {
                "question": "What is...",
                "answer": "The answer is...",
                "contexts": ["context string 1", "context string 2"],
                "ground_truth": "The true answer is..."
            }, ...
        ]
        """
        logger.info(f"Preparing {len(evaluation_data)} samples for RAGAS evaluation...")
        
        # Convert list of dicts to the HuggingFace Dataset format expected by RAGAS
        data_dict = {
            "question": [item["question"] for item in evaluation_data],
            "answer": [item["answer"] for item in evaluation_data],
            "contexts": [item["contexts"] for item in evaluation_data],
            "ground_truth": [item["ground_truth"] for item in evaluation_data],
        }
        
        hf_dataset = Dataset.from_dict(data_dict)
        
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ]
        
        logger.info("Running RAGAS evaluation (this may take a while)...")
        
        # If custom llm/embeddings are provided, pass them to evaluate
        eval_kwargs = {"dataset": hf_dataset, "metrics": metrics}
        if self.llm:
            eval_kwargs["llm"] = self.llm
        if self.embeddings:
            eval_kwargs["embeddings"] = self.embeddings
            
        result = evaluate(**eval_kwargs)
        
        # Extract aggregate scores
        scores = {
            "faithfulness": result.get("faithfulness", 0.0),
            "answer_relevancy": result.get("answer_relevancy", 0.0),
            "context_precision": result.get("context_precision", 0.0),
            "context_recall": result.get("context_recall", 0.0),
        }
        
        logger.info("Evaluation complete. Checking against targets...")
        passed_all = True
        for metric, score in scores.items():
            target = self.targets.get(metric, 0.0)
            status = "PASS" if score >= target else "FAIL"
            if status == "FAIL":
                passed_all = False
            logger.info(f"{metric}: {score:.4f} (Target: {target}) -> {status}")
            
        return {
            "aggregate_scores": scores,
            "passed": passed_all,
            "targets": self.targets,
            "detailed_results": result.to_pandas().to_dict(orient="records")
        }
