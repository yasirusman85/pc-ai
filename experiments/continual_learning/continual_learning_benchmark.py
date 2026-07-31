"""
continual_learning_benchmark.py — Test CRF vs Transformer on continual learning
===========================================================================
This is where CRF can actually beat Transformers.

Key hypothesis: CRF cells can specialize and persist, avoiding catastrophic forgetting
that plagues Transformers.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
import json
from pathlib import Path

from crf_truly_batched import TrulyBatchedCRFLanguageModel
from transformer import GPT
from data import ArithmeticDataset, get_datasets, make_dataloader


class ContinualLearningBenchmark:
    """
    Benchmark for continual learning capability.
    
    Tasks:
    1. Task A: Simple arithmetic (2 operations)
    2. Task B: Complex arithmetic (4 operations)
    3. Task C: Code completion (simple)
    
    Measure: How well does model retain task A after learning B and C?
    """
    
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.results = {}
    
    def create_task_datasets(self):
        """Create datasets for different tasks."""
        print("Creating task datasets...")
        
        # Task A: Simple arithmetic (easy)
        task_a_train = ArithmeticDataset(n_samples=1000, seq_len=32, difficulty='easy')
        task_a_test = ArithmeticDataset(n_samples=200, seq_len=32, difficulty='easy')
        
        # Task B: Complex arithmetic (hard)
        task_b_train = ArithmeticDataset(n_samples=1000, seq_len=48, difficulty='hard')
        task_b_test = ArithmeticDataset(n_samples=200, seq_len=48, difficulty='hard')
        
        # Task C: Synthetic language (different task)
        task_c_train = ArithmeticDataset(n_samples=1000, seq_len=64, difficulty='mixed')
        task_c_test = ArithmeticDataset(n_samples=200, seq_len=64, difficulty='mixed')
        
        tasks = {
            'task_a': {
                'name': 'Simple Arithmetic',
                'train': task_a_train,
                'test': task_a_test,
            },
            'task_b': {
                'name': 'Complex Arithmetic', 
                'train': task_b_train,
                'test': task_b_test,
            },
            'task_c': {
                'name': 'Mixed Arithmetic',
                'train': task_c_train,
                'test': task_c_test,
            }
        }
        
        return tasks
    
    def train_on_task(self, model, task_data, n_epochs=5, lr=3e-4):
        """Train model on a single task."""
        model.train()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        
        train_loader = make_dataloader(task_data['train'], batch_size=16)
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                
                if hasattr(model, 'crf'):
                    result = model(x, targets=y)
                    if len(result) == 3:
                        logits, loss, _ = result
                    else:
                        logits, loss = result
                else:
                    logits, loss = model(x, targets=y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 2 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}: Loss={total_loss/len(train_loader):.4f}")
    
    def evaluate_on_task(self, model, task_data):
        """Evaluate model on a task."""
        model.eval()
        
        test_loader = make_dataloader(task_data['test'], batch_size=16, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                if hasattr(model, 'crf'):
                    result = model(x, targets=y)
                    if len(result) == 3:
                        logits, _, _ = result
                    else:
                        logits, _ = result
                else:
                    logits, _ = model(x)
                
                predictions = logits.argmax(dim=-1)
                correct += (predictions == y).sum().item()
                total += y.numel()
        
        accuracy = correct / total
        return accuracy
    
    def run_continual_learning_test(self, model_name, model_class, model_kwargs):
        """
        Run continual learning test for a model.
        
        Sequence:
        1. Learn task A
        2. Evaluate on A
        3. Learn task B
        4. Evaluate on A and B
        5. Learn task C
        6. Evaluate on A, B, and C
        
        Key metric: How much does performance on A degrade after learning B and C?
        """
        print(f"\n{'='*80}")
        print(f"Testing {model_name}")
        print(f"{'='*80}")
        
        # Create fresh model
        model = model_class(**model_kwargs).to(self.device)
        
        # Create tasks
        tasks = self.create_task_datasets()
        
        results = {
            'model_name': model_name,
            'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'stages': {}
        }
        
        # Stage 1: Learn task A
        print("\nStage 1: Learning Task A (Simple Arithmetic)")
        self.train_on_task(model, tasks['task_a'], n_epochs=5)
        acc_a_1 = self.evaluate_on_task(model, tasks['task_a'])
        print(f"  Task A accuracy: {acc_a_1:.3f}")
        results['stages']['after_task_a'] = {
            'task_a': acc_a_1,
        }
        
        # Stage 2: Learn task B
        print("\nStage 2: Learning Task B (Complex Arithmetic)")
        self.train_on_task(model, tasks['task_b'], n_epochs=5)
        acc_a_2 = self.evaluate_on_task(model, tasks['task_a'])
        acc_b_2 = self.evaluate_on_task(model, tasks['task_b'])
        print(f"  Task A accuracy: {acc_a_2:.3f} (degradation: {(acc_a_1 - acc_a_2)/acc_a_1*100:.1f}%)")
        print(f"  Task B accuracy: {acc_b_2:.3f}")
        results['stages']['after_task_b'] = {
            'task_a': acc_a_2,
            'task_b': acc_b_2,
            'task_a_degradation': (acc_a_1 - acc_a_2) / acc_a_1,
        }
        
        # Stage 3: Learn task C
        print("\nStage 3: Learning Task C (Code Completion)")
        self.train_on_task(model, tasks['task_c'], n_epochs=5)
        acc_a_3 = self.evaluate_on_task(model, tasks['task_a'])
        acc_b_3 = self.evaluate_on_task(model, tasks['task_b'])
        acc_c_3 = self.evaluate_on_task(model, tasks['task_c'])
        print(f"  Task A accuracy: {acc_a_3:.3f} (degradation: {(acc_a_1 - acc_a_3)/acc_a_1*100:.1f}%)")
        print(f"  Task B accuracy: {acc_b_3:.3f} (degradation: {(acc_b_2 - acc_b_3)/acc_b_2*100:.1f}%)")
        print(f"  Task C accuracy: {acc_c_3:.3f}")
        results['stages']['after_task_c'] = {
            'task_a': acc_a_3,
            'task_b': acc_b_3,
            'task_c': acc_c_3,
            'task_a_degradation': (acc_a_1 - acc_a_3) / acc_a_1,
            'task_b_degradation': (acc_b_2 - acc_b_3) / acc_b_2,
        }
        
        # Calculate forgetting metrics
        results['forgetting_metrics'] = {
            'task_a_forgetting': (acc_a_1 - acc_a_3) / acc_a_1,
            'task_b_forgetting': (acc_b_2 - acc_b_3) / acc_b_2,
            'total_forgetting': ((acc_a_1 - acc_a_3) + (acc_b_2 - acc_b_3)) / (acc_a_1 + acc_b_2),
        }
        
        return results
    
    def run_comparison(self):
        """Run comparison between CRF and Transformer."""
        vocab_size = 99
        d_model = 64
        
        results = []
        
        # Test CRF
        crf_results = self.run_continual_learning_test(
            "CRF",
            TrulyBatchedCRFLanguageModel,
            {
                'vocab_size': vocab_size,
                'd_model': d_model,
                'n_heads': 4,
                'n_layers': 2,
            }
        )
        results.append(crf_results)
        
        # Test Transformer
        transformer_results = self.run_continual_learning_test(
            "Transformer",
            GPT,
            {
                'vocab_size': vocab_size,
                'd_model': d_model,
                'n_heads': 4,
                'n_layers': 2,
                'max_seq_len': 64,
            }
        )
        results.append(transformer_results)
        
        # Print comparison
        print(f"\n{'='*80}")
        print("CONTINUAL LEARNING COMPARISON")
        print(f"{'='*80}")
        
        for result in results:
            print(f"\n{result['model_name']}:")
            print(f"  Parameters: {result['n_params']:,}")
            print(f"  Task A initial accuracy: {result['stages']['after_task_a']['task_a']:.3f}")
            print(f"  Task A final accuracy: {result['stages']['after_task_c']['task_a']:.3f}")
            print(f"  Task A forgetting: {result['forgetting_metrics']['task_a_forgetting']*100:.1f}%")
            print(f"  Task B forgetting: {result['forgetting_metrics']['task_b_forgetting']*100:.1f}%")
            print(f"  Total forgetting: {result['forgetting_metrics']['total_forgetting']*100:.1f}%")
        
        # Compare forgetting
        crf_forgetting = results[0]['forgetting_metrics']['total_forgetting']
        transformer_forgetting = results[1]['forgetting_metrics']['total_forgetting']
        
        print(f"\nForgetting Comparison:")
        print(f"  CRF: {crf_forgetting*100:.1f}%")
        print(f"  Transformer: {transformer_forgetting*100:.1f}%")
        print(f"  CRF advantage: {(transformer_forgetting - crf_forgetting)*100:.1f}%")
        
        # Save results
        output_file = Path("continual_learning_results.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
        return results


if __name__ == "__main__":
    benchmark = ContinualLearningBenchmark(device='cpu')
    results = benchmark.run_comparison()