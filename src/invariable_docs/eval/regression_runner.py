"""
Regression Evaluation Runner.

CLI harness to execute the RAGAS evaluator over the golden dataset
and output historical regression reports.
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from invariable_docs.eval.ragas_evaluator import RagasEvaluator

# In a real environment, we would import our pipelines and run them on the fly.
# For demonstration in this module, we simulate the execution step or assume
# the user provides the pre-generated outputs.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_golden_dataset(path: str) -> List[Dict[str, Any]]:
    """Load the golden dataset from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simulate_pipeline_execution(golden_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simulate running our RAG pipeline on the golden dataset.
    In reality, we would call HybridRetrievalEngine and GenerationEngine here.
    """
    logger.info("Executing RAG Pipeline on Golden Dataset (Simulation)...")
    eval_payload = []
    
    for item in golden_data:
        # SIMULATION: We just echo the ground truth as the answer for testing the evaluator.
        # In production:
        # chunks = hybrid_engine.retrieve(item["question"])
        # answer = generation_engine.generate_answer(item["question"], chunks)
        
        eval_payload.append({
            "question": item["question"],
            "answer": item["ground_truth_answer"],  # Perfect answer
            "contexts": item["ground_truth_contexts"], # Perfect contexts
            "ground_truth": item["ground_truth_answer"],
        })
        
    return eval_payload


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS Regression Evaluation")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="src/invariable_docs/eval/golden_dataset.json",
        help="Path to the golden dataset JSON file."
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="eval_results",
        help="Directory to save the regression report."
    )
    args = parser.parse_args()

    logger.info(f"Starting RAGAS Regression Evaluation using dataset: {args.dataset}")
    
    # 1. Load Data
    golden_data = load_golden_dataset(args.dataset)
    
    # 2. Run Pipeline (simulated here for framework setup)
    eval_payload = simulate_pipeline_execution(golden_data)
    
    # 3. Evaluate
    evaluator = RagasEvaluator()
    results = evaluator.evaluate_dataset(eval_payload)
    
    # 4. Save Report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"run_{timestamp}.json"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Saved regression report to {report_path}")
    
    if not results["passed"]:
        logger.error("Regression Test FAILED! One or more metrics missed the target thresholds.")
        exit(1)
    else:
        logger.info("Regression Test PASSED! All metrics met target thresholds.")
        exit(0)


if __name__ == "__main__":
    main()
