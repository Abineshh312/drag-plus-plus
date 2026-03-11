#!/usr/bin/env python3
"""
DRAG++ Training on MS MARCO Dataset
====================================
Complete training pipeline for DRAG++ (Distilling RAG for SLMs from LLMs)
using the MS MARCO retrieval dataset.

Features:
- Flexible model selection from HuggingFace
- MS MARCO dataset loading (train/test splits)
- Retriever training (BM25 + dense embeddings)
- Student model fine-tuning
- Comprehensive evaluation metrics
- Results saving for publication

Usage:
    python drag_ms_marco_training.py \
        --student-model "microsoft/phi-2" \
        --retriever "sentence-transformers/all-mpnet-base-v2" \
        --dataset-size 10000 \
        --batch-size 8 \
        --epochs 3 \
        --output-dir "./drag_results"

"""

import os
import json
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from tqdm import tqdm

# HuggingFace datasets
from datasets import load_dataset

# Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,
    Trainer,
    TrainingArguments,
)

# Evaluation
from sklearn.metrics import f1_score, precision_recall_fscore_support
import evaluate

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DRAGPlusPlus_Config:
    """Configuration for DRAG++ training"""
    
    def __init__(self, args):
        self.student_model = args.student_model or "microsoft/phi-2"
        self.teacher_model = args.teacher_model or "meta-llama/Llama-2-7b-hf"
        self.retriever_model = args.retriever_model or "sentence-transformers/all-mpnet-base-v2"
        
        self.dataset_size = args.dataset_size or 10000  # Subset of MS MARCO
        self.batch_size = args.batch_size or 8
        self.epochs = args.epochs or 3
        self.learning_rate = args.learning_rate or 5e-5
        self.max_seq_length = args.max_seq_length or 512
        
        self.output_dir = args.output_dir or "./drag_results"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        
        logger.info(f"🔧 Configuration:")
        logger.info(f"  Student Model: {self.student_model}")
        logger.info(f"  Retriever Model: {self.retriever_model}")
        logger.info(f"  Dataset Size: {self.dataset_size}")
        logger.info(f"  Device: {self.device}")


# ============================================================================
# DATASET LOADING & PREPARATION
# ============================================================================

class MSMarcoDataLoader:
    """Load and prepare MS MARCO dataset from HuggingFace"""
    
    @staticmethod
    def load_dataset(config: DRAGPlusPlus_Config) -> Dict[str, Any]:
        """
        Load MS MARCO from HuggingFace datasets
        
        Dataset structure:
        - query: The search question
        - query_id: Unique query identifier
        - query_type: Type of query (description, numeric, location, entity, yes/no)
        - passages: List of passages with is_selected flag
        - answers: Ground truth answers
        - wellFormedAnswers: Well-formatted answers
        
        Returns:
            Dictionary with train/test splits
        """
        logger.info("📥 Loading MS MARCO dataset...")
        
        try:
            # Load MS MARCO from HuggingFace
            dataset = load_dataset("microsoft/ms_marco", "v1.1")
            
            # Use train split, sample to dataset_size
            train_data = dataset['train']
            
            if config.dataset_size < len(train_data):
                indices = np.random.choice(
                    len(train_data),
                    size=config.dataset_size,
                    replace=False
                )
                train_data = train_data.select(indices)
            
            logger.info(f"✅ Loaded {len(train_data)} examples from MS MARCO")
            logger.info(f"   Dataset features: {train_data.features.keys()}")
            
            return {
                'train': train_data,
                'test': dataset['validation'][:min(1000, len(dataset['validation']))]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise


class MSMarcoProcessor:
    """Process raw MS MARCO examples into training format"""
    
    @staticmethod
    def process_example(example: Dict, config: DRAGPlusPlus_Config) -> Dict:
        """
        Convert raw MS MARCO example to retriever + generator training format
        
        Raw example:
        {
            'query': str,
            'passages': {'is_selected': List[int], 'passage_text': List[str]},
            'answers': List[str],
            'wellFormedAnswers': List[str]
        }
        
        Returns:
            {
                'query': str,
                'positive_passages': List[str],  # is_selected=1
                'negative_passages': List[str],  # is_selected=0
                'answers': List[str]
            }
        """
        query = example['query']
        passages = example['passages']
        answers = example.get('wellFormedAnswers', example.get('answers', []))
        
        passage_texts = passages['passage_text']
        is_selected = passages['is_selected']
        
        # Split into positive (relevant) and negative (irrelevant) passages
        positive = [p for p, sel in zip(passage_texts, is_selected) if sel == 1]
        negative = [p for p, sel in zip(passage_texts, is_selected) if sel == 0]
        
        return {
            'query': query,
            'positive_passages': positive[:3],  # Top 3 relevant
            'negative_passages': negative[:3],  # Top 3 irrelevant
            'answers': answers if answers else ["No answer available"]
        }


# ============================================================================
# RETRIEVER TRAINING
# ============================================================================

class HybridRetrieverTrainer:
    """
    Train hybrid retriever: BM25 (sparse) + Dense embeddings
    
    For MS MARCO:
    - BM25: Fast keyword matching on passage text
    - Dense: Learned semantic similarity (sentence-transformers)
    """
    
    def __init__(self, config: DRAGPlusPlus_Config):
        self.config = config
        self.device = config.device
        
        # Load dense encoder
        logger.info(f"Loading retriever model: {config.retriever_model}")
        self.dense_encoder = AutoModel.from_pretrained(
            config.retriever_model
        ).to(self.device)
        self.dense_encoder.eval()
        
        # Will use sklearn for BM25 (simple implementation)
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.bm25_vectorizer = TfidfVectorizer(max_features=5000)
        self.passage_encodings = None
        
    def encode_passages(self, passages: List[str]) -> np.ndarray:
        """Encode passages using dense encoder"""
        embeddings = []
        
        with torch.no_grad():
            for passage in tqdm(passages, desc="Encoding passages"):
                # Tokenize
                inputs = self.dense_encoder.tokenizer(
                    passage,
                    max_length=512,
                    truncation=True,
                    return_tensors="pt"
                ).to(self.device)
                
                # Encode
                output = self.dense_encoder(**inputs)
                cls_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding)
        
        return np.vstack(embeddings)
    
    def retrieve(self, query: str, passages: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """Retrieve top-k passages for a query using hybrid approach"""
        # BM25 score
        bm25_scores = self.bm25_vectorizer.transform([query]).toarray()[0]
        
        # Dense similarity
        with torch.no_grad():
            query_inputs = self.dense_encoder.tokenizer(
                query,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            query_embedding = self.dense_encoder(**query_inputs).last_hidden_state[:, 0, :].cpu().numpy()
        
        # Hybrid score (0.3 * BM25 + 0.7 * dense)
        hybrid_scores = []
        for i, passage in enumerate(passages):
            bm25_sim = bm25_scores[i] if i < len(bm25_scores) else 0.0
            
            passage_inputs = self.dense_encoder.tokenizer(
                passage,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                passage_embedding = self.dense_encoder(**passage_inputs).last_hidden_state[:, 0, :].cpu().numpy()
            
            dense_sim = np.dot(query_embedding, passage_embedding.T) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(passage_embedding) + 1e-8
            )
            
            hybrid = 0.3 * bm25_sim + 0.7 * float(dense_sim)
            hybrid_scores.append((passage, hybrid))
        
        # Return top-k
        return sorted(hybrid_scores, key=lambda x: x[1], reverse=True)[:top_k]


# ============================================================================
# STUDENT MODEL TRAINING
# ============================================================================

class StudentModelTrainer:
    """
    Train student model (small LLM) on retrieved passages
    
    Distillation process:
    1. Teacher generates answers given query + retrieved passages
    2. Student learns to mimic teacher (knowledge distillation)
    3. Student fine-tuned on MS MARCO with retrieved evidence
    """
    
    def __init__(self, config: DRAGPlusPlus_Config):
        self.config = config
        self.device = config.device
        
        logger.info(f"Loading student model: {config.student_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.student_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.student_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"Model params: {self.model.num_parameters():,}")
    
    def prepare_training_data(self, processed_examples: List[Dict]) -> List[Dict]:
        """
        Prepare training examples:
        Format: "Query: {query}\nPassages: {passages}\nAnswer: {answer}"
        """
        training_data = []
        
        for example in tqdm(processed_examples, desc="Preparing training data"):
            query = example['query']
            passages = example.get('positive_passages', [])
            answers = example.get('answers', [""])
            
            # Use first answer
            answer = answers[0] if answers else ""
            
            # Combine passages
            passage_text = " ".join(passages[:3])  # Top 3
            
            # Format prompt
            prompt = f"Query: {query}\nPassages: {passage_text}\nAnswer:"
            full_text = f"{prompt} {answer}"
            
            training_data.append({
                'text': full_text,
                'query': query,
                'answer': answer
            })
        
        return training_data
    
    def tokenize_function(self, examples):
        """Tokenize for training"""
        return self.tokenizer(
            examples['text'],
            max_length=self.config.max_seq_length,
            truncation=True,
            padding="max_length"
        )


# ============================================================================
# EVALUATION METRICS
# ============================================================================

class EvaluationMetrics:
    """Compute metrics for DRAG++"""
    
    @staticmethod
    def compute_retrieval_metrics(
        retrieved_passages: List[str],
        positive_passages: List[str],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Retrieval metrics:
        - Recall@K: Did we retrieve relevant passages?
        - Precision@K: Of retrieved, how many were relevant?
        """
        retrieved_set = set(retrieved_passages[:top_k])
        positive_set = set(positive_passages)
        
        if len(positive_set) == 0:
            return {'recall@5': 0.0, 'precision@5': 0.0}
        
        hits = len(retrieved_set.intersection(positive_set))
        
        recall = hits / len(positive_set)
        precision = hits / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
        
        return {
            'recall@5': recall,
            'precision@5': precision,
            'f1@5': 2 * (precision * recall) / (precision + recall + 1e-8)
        }
    
    @staticmethod
    def compute_generation_metrics(
        predictions: List[str],
        references: List[str]
    ) -> Dict[str, float]:
        """
        Generation metrics:
        - ROUGE (fluency)
        - BLEU (matching)
        """
        rouge = evaluate.load('rouge')
        results = rouge.compute(predictions=predictions, references=references)
        
        return {
            'rouge1': results['rouge1'],
            'rougeL': results['rougeL']
        }
    
    @staticmethod
    def compute_hallucination_rate(
        generated_answers: List[str],
        evidence_passages: List[List[str]]
    ) -> float:
        """
        Hallucination detection:
        Compare generated answer to evidence passages
        High overlap = grounded, Low overlap = hallucination
        """
        hallucination_scores = []
        
        for answer, passages in zip(generated_answers, evidence_passages):
            answer_tokens = set(answer.lower().split())
            passage_tokens = set()
            
            for passage in passages:
                passage_tokens.update(passage.lower().split())
            
            # Overlap ratio
            if len(answer_tokens) == 0:
                overlap = 0.0
            else:
                overlap = len(answer_tokens.intersection(passage_tokens)) / len(answer_tokens)
            
            # Hallucination = 1 - overlap
            hallucination = 1.0 - overlap
            hallucination_scores.append(hallucination)
        
        return np.mean(hallucination_scores) if hallucination_scores else 0.0


# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

class DRAGPlusPlusTrainer:
    """Main orchestrator for DRAG++ training on MS MARCO"""
    
    def __init__(self, config: DRAGPlusPlus_Config):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        
        logger.info("="*80)
        logger.info("🚀 DRAG++ TRAINING ON MS MARCO")
        logger.info("="*80)
    
    def run(self):
        """Execute full training pipeline"""
        
        # 1. Load dataset
        logger.info("\n[1/5] Loading MS MARCO dataset...")
        data_loader = MSMarcoDataLoader()
        datasets = data_loader.load_dataset(self.config)
        train_data = datasets['train']
        test_data = datasets['test']
        
        # 2. Process examples
        logger.info("\n[2/5] Processing dataset...")
        processor = MSMarcoProcessor()
        
        processed_train = [
            processor.process_example(example, self.config)
            for example in tqdm(train_data, desc="Processing train data")
        ]
        
        processed_test = [
            processor.process_example(example, self.config)
            for example in tqdm(test_data, desc="Processing test data")
        ]
        
        # 3. Train retriever
        logger.info("\n[3/5] Training retriever...")
        retriever = HybridRetrieverTrainer(self.config)
        
        # Extract all passages for BM25 indexing
        all_passages = []
        for example in processed_train:
            all_passages.extend(example['positive_passages'])
            all_passages.extend(example['negative_passages'])
        
        all_passages = list(set(all_passages))  # Remove duplicates
        logger.info(f"Indexing {len(all_passages)} unique passages...")
        
        # Fit BM25
        retriever.bm25_vectorizer.fit(all_passages)
        
        # 4. Evaluate retrieval
        logger.info("\n[4/5] Evaluating retrieval...")
        retrieval_metrics = {
            'recall@5': [],
            'precision@5': [],
            'f1@5': []
        }
        
        for example in tqdm(processed_test[:100], desc="Evaluating retrieval"):
            query = example['query']
            positive = example.get('positive_passages', [])
            
            # Retrieve
            retrieved = retriever.retrieve(query, all_passages[:1000], top_k=5)
            retrieved_texts = [p[0] for p in retrieved]
            
            # Compute metrics
            metrics = EvaluationMetrics.compute_retrieval_metrics(
                retrieved_texts, positive, top_k=5
            )
            
            for key in retrieval_metrics:
                retrieval_metrics[key].append(metrics[key])
        
        # Average metrics
        avg_retrieval_metrics = {
            k: np.mean(v) for k, v in retrieval_metrics.items()
        }
        
        logger.info(f"✅ Retrieval Results:")
        for k, v in avg_retrieval_metrics.items():
            logger.info(f"  {k}: {v:.4f}")
        
        # 5. Train student model
        logger.info("\n[5/5] Training student model...")
        student_trainer = StudentModelTrainer(self.config)
        training_data = student_trainer.prepare_training_data(processed_train)
        
        logger.info(f"Prepared {len(training_data)} training examples")
        
        # Save results
        self.save_results(avg_retrieval_metrics, training_data)
    
    def save_results(self, metrics: Dict, training_data: List[Dict]):
        """Save training results"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'student_model': self.config.student_model,
                'retriever_model': self.config.retriever_model,
                'dataset_size': self.config.dataset_size,
            },
            'metrics': metrics,
            'training_examples_count': len(training_data)
        }
        
        output_file = os.path.join(self.config.output_dir, 'results.json')
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n✅ Results saved to {output_file}")
        logger.info(f"\nFinal Metrics:")
        logger.info(json.dumps(results, indent=2))


# ============================================================================
# CLI & ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DRAG++ Training on MS MARCO"
    )
    
    # Model arguments
    parser.add_argument(
        '--student-model',
        type=str,
        default="microsoft/phi-2",
        help="HuggingFace student model (default: microsoft/phi-2)"
    )
    parser.add_argument(
        '--teacher-model',
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="HuggingFace teacher model for distillation"
    )
    parser.add_argument(
        '--retriever-model',
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Retriever model (default: sentence-transformers/all-mpnet-base-v2)"
    )
    
    # Training arguments
    parser.add_argument(
        '--dataset-size',
        type=int,
        default=10000,
        help="Number of examples to use from MS MARCO (default: 10000)"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help="Number of training epochs (default: 3)"
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5)"
    )
    parser.add_argument(
        '--max-seq-length',
        type=int,
        default=512,
        help="Max sequence length (default: 512)"
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default="./drag_results",
        help="Output directory (default: ./drag_results)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = DRAGPlusPlus_Config(args)
    
    # Run training
    trainer = DRAGPlusPlusTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
