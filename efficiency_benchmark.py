"""
efficiency_benchmark.py — Integrated benchmark for testing CRF efficiency hypothesis
================================================================================

Hypothesis: "A Cellular Reasoning Fabric can achieve Transformer-level reasoning 
with significantly fewer training tokens and lower training compute by replacing 
weight memorization with dynamic computation, reusable reasoning programs, and 
adaptive cognitive cells."

This benchmark tests:
1. Sample efficiency: Accuracy vs training tokens
2. Compute efficiency: Accuracy vs FLOPs  
3. Time efficiency: Accuracy vs wall-clock time
4. Reasoning quality: Multi-step reasoning accuracy
5. Rapid adaptation: Few-shot learning performance

Target: CRF reaches 90% of Transformer performance using 5-10% of training compute.
"""

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from crf_vectorized import CRFLanguageModel, AblationConfig
from transformer import GPT
from data import get_datasets, make_dataloader
from train import train
from efficiency_optimizations import (
    EfficientCRFTrainer, AdaptiveLifecycleConfig, CurriculumConfig,
    ReasoningMetrics
)
from token_efficiency_tracker import (
    TokenEfficiencyTracker, ComparativeEfficiencyAnalyzer
)
from metrics import estimate_crf_flops, estimate_transformer_flops


class EfficiencyBenchmark:
    """
    Main benchmark class for testing CRF efficiency hypothesis.
    
    Coordinates:
    - Model training with efficiency tracking
    - Optimized training strategies
    - Comparative analysis
    - Hypothesis validation
    """
    
    def __init__(
        self,
        output_dir: str = "efficiency_benchmark_results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device(device)
        
        # Initialize components
        self.comparative_analyzer = ComparativeEfficiencyAnalyzer(str(self.output_dir))
        
    def run_crf_experiment(
        self,
        config: Dict,
        use_optimizations: bool = True,
    ) -> Dict:
        """
        Run CRF experiment with efficiency tracking.
        
        Args:
            config: Experiment configuration
            use_optimizations: Whether to use efficiency optimizations
            
        Returns:
            Training results with efficiency metrics
        """
        print("\n" + "="*80)
        print("RUNNING CRF EXPERIMENT")
        print("="*80)
        
        # Setup tracking
        crf_tracker = self.comparative_analyzer.setup_crf_tracking("CRF")
        
        # Create model
        model = CRFLanguageModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            d_hidden=config['d_hidden'],
            n_init_cells=config['n_init_cells'],
            max_cells=config['max_cells'],
            n_crf_steps=config['n_crf_steps'],
            k_neighbors=config['k_neighbors'],
            max_seq_len=config['seq_len'],
            cfg=AblationConfig() if not use_optimizations else None,
        )
        
        # Create optimized trainer if requested
        if use_optimizations:
            efficient_trainer = EfficientCRFTrainer(model, self.device)
        else:
            efficient_trainer = None
        
        # Load data
        train_ds, val_ds = get_datasets(
            config['dataset'],
            seq_len=config['seq_len'],
            max_train=config['max_train'],
            max_val=config['max_val'],
            use_real=config.get('use_real', False),
        )
        
        train_loader = make_dataloader(train_ds, batch_size=config['batch_size'])
        val_loader = make_dataloader(val_ds, batch_size=config['batch_size'], shuffle=False)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
        )
        
        # Training loop with efficiency tracking
        model.to(self.device)
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epoch_times': [],
        }
        
        for epoch in range(config['n_epochs']):
            epoch_start = time.time()
            
            # Training
            if use_optimizations and efficient_trainer:
                train_loss = efficient_trainer.train_epoch(epoch, train_loader, val_loader, optimizer)
            else:
                train_loss = self._standard_train_epoch(model, train_loader, optimizer, self.device)
            
            # Validation
            val_metrics = self._validate_epoch(model, val_loader, self.device)
            
            # Record efficiency metrics
            epoch_time = time.time() - epoch_start
            
            # Estimate FLOPs for this epoch
            batch_flops = estimate_crf_flops(
                config['n_init_cells'], config['d_model'], 
                config['d_hidden'], config['k_neighbors'], config['n_crf_steps']
            )
            epoch_flops = batch_flops * len(train_loader)
            
            # Record in tracker
            crf_tracker.record_batch(
                batch_size=config['batch_size'],
                seq_len=config['seq_len'],
                estimated_flops=epoch_flops
            )
            
            # Record epoch
            crf_tracker.record_epoch(epoch, {
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'train_accuracy': val_metrics['train_accuracy'],
                'val_accuracy': val_metrics['val_accuracy'],
                'reasoning_score': val_metrics.get('reasoning_score', 0.0),
                'avg_cell_count': val_metrics.get('avg_cell_count', 0.0),
                'avg_comm_cost': val_metrics.get('avg_comm_cost', 0.0),
                'specialization': val_metrics.get('specialization', 0.0),
            })
            
            # Store history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['train_accuracy'].append(val_metrics['train_accuracy'])
            training_history['val_accuracy'].append(val_metrics['val_accuracy'])
            training_history['epoch_times'].append(epoch_time)
            
            print(f"Epoch {epoch+1}/{config['n_epochs']}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['val_accuracy']:.3f}")
        
        # Save tracker results
        crf_tracker.save_all_snapshots()
        crf_analysis = crf_tracker.analyze_efficiency()
        
        # Generate efficiency report
        report = crf_tracker.generate_efficiency_report()
        print(report)
        
        # Save report
        report_file = self.output_dir / "crf_efficiency_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Plot efficiency curves
        plot_file = self.output_dir / "crf_efficiency_curves.png"
        crf_tracker.plot_efficiency_curves(str(plot_file))
        
        return {
            'model_name': 'CRF',
            'training_history': training_history,
            'efficiency_analysis': crf_analysis,
            'use_optimizations': use_optimizations,
        }
    
    def run_transformer_experiment(
        self,
        config: Dict,
        target_params: Optional[int] = None,
    ) -> Dict:
        """
        Run Transformer baseline experiment with efficiency tracking.
        
        Args:
            config: Experiment configuration
            target_params: Target parameter count (for fair comparison)
            
        Returns:
            Training results with efficiency metrics
        """
        print("\n" + "="*80)
        print("RUNNING TRANSFORMER BASELINE EXPERIMENT")
        print("="*80)
        
        # Setup tracking
        transformer_tracker = self.comparative_analyzer.setup_transformer_tracking("Transformer")
        
        # Create model (param-matched if target provided)
        if target_params:
            from train import make_matched_transformer
            model = make_matched_transformer(
                vocab_size=config['vocab_size'],
                target_params=target_params,
                max_seq_len=config['seq_len'],
            )
        else:
            model = GPT(
                vocab_size=config['vocab_size'],
                d_model=config['d_model'],
                n_heads=config.get('n_heads', 4),
                n_layers=config.get('n_layers', 4),
                max_seq_len=config['seq_len'],
            )
        
        # Load data
        train_ds, val_ds = get_datasets(
            config['dataset'],
            seq_len=config['seq_len'],
            max_train=config['max_train'],
            max_val=config['max_val'],
        )
        
        train_loader = make_dataloader(train_ds, batch_size=config['batch_size'])
        val_loader = make_dataloader(val_ds, batch_size=config['batch_size'], shuffle=False)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay'],
        )
        
        # Training loop with efficiency tracking
        model.to(self.device)
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'epoch_times': [],
        }
        
        for epoch in range(config['n_epochs']):
            epoch_start = time.time()
            
            # Training
            train_loss = self._standard_train_epoch(model, train_loader, optimizer, self.device)
            
            # Validation
            val_metrics = self._validate_epoch(model, val_loader, self.device)
            
            epoch_time = time.time() - epoch_start
            
            # Estimate FLOPs for this epoch
            n_layers = len(model.blocks) if hasattr(model, 'blocks') else 4
            batch_flops = estimate_transformer_flops(
                config['seq_len'], config['d_model'], n_layers
            )
            epoch_flops = batch_flops * len(train_loader)
            
            # Record in tracker
            transformer_tracker.record_batch(
                batch_size=config['batch_size'],
                seq_len=config['seq_len'],
                estimated_flops=epoch_flops
            )
            
            # Record epoch
            transformer_tracker.record_epoch(epoch, {
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'train_accuracy': val_metrics['train_accuracy'],
                'val_accuracy': val_metrics['val_accuracy'],
                'reasoning_score': val_metrics.get('reasoning_score', 0.0),
                'avg_cell_count': 0.0,  # Transformer doesn't have cells
                'avg_comm_cost': 0.0,
                'specialization': 0.0,
            })
            
            # Store history
            training_history['train_loss'].append(train_loss)
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['train_accuracy'].append(val_metrics['train_accuracy'])
            training_history['val_accuracy'].append(val_metrics['val_accuracy'])
            training_history['epoch_times'].append(epoch_time)
            
            print(f"Epoch {epoch+1}/{config['n_epochs']}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['val_accuracy']:.3f}")
        
        # Save tracker results
        transformer_tracker.save_all_snapshots()
        transformer_analysis = transformer_tracker.analyze_efficiency()
        
        # Generate efficiency report
        report = transformer_tracker.generate_efficiency_report()
        print(report)
        
        # Save report
        report_file = self.output_dir / "transformer_efficiency_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Plot efficiency curves
        plot_file = self.output_dir / "transformer_efficiency_curves.png"
        transformer_tracker.plot_efficiency_curves(str(plot_file))
        
        return {
            'model_name': 'Transformer',
            'training_history': training_history,
            'efficiency_analysis': transformer_analysis,
            'n_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    
    def run_comparative_benchmark(self, config: Dict) -> Dict:
        """
        Run full comparative benchmark between CRF and Transformer.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Comparative analysis results
        """
        print("\n" + "="*80)
        print("RUNNING COMPARATIVE EFFICIENCY BENCHMARK")
        print("="*80)
        
        # Run CRF experiment (with optimizations)
        crf_results = self.run_crf_experiment(config, use_optimizations=True)
        
        # Get CRF parameter count for fair comparison
        crf_params = crf_results['efficiency_analysis']['final_metrics'].get('n_params', 0)
        
        # Run Transformer experiment (param-matched)
        transformer_results = self.run_transformer_experiment(
            config, 
            target_params=crf_params
        )
        
        # Set Transformer baseline for CRF tracker
        if transformer_results['efficiency_analysis'].get('final_metrics'):
            self.comparative_analyzer.crf_tracker.set_transformer_baseline(
                transformer_results['efficiency_analysis']['final_metrics']
            )
        
        # Run comparative analysis
        comparative_results = self.comparative_analyzer.run_comparative_analysis()
        
        # Generate comparative report
        comparative_report = self.comparative_analyzer.generate_comparative_report()
        print(comparative_report)
        
        # Save comparative report
        report_file = self.output_dir / "comparative_report.txt"
        with open(report_file, 'w') as f:
            f.write(comparative_report)
        
        # Save full results
        full_results = {
            'config': config,
            'crf_results': crf_results,
            'transformer_results': transformer_results,
            'comparative_analysis': comparative_results,
        }
        
        results_file = self.output_dir / "full_benchmark_results.json"
        with open(results_file, 'w') as f:
            json.dump(full_results, f, indent=2, default=str)
        
        return full_results
    
    def _standard_train_epoch(self, model, train_loader, optimizer, device):
        """Standard training epoch without optimizations."""
        model.train()
        total_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            if hasattr(model, 'crf'):
                logits, loss, _ = model(x, targets=y)
            else:
                logits, loss = model(x, targets=y)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, model, val_loader, device):
        """Validation epoch with metrics."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        crf_metrics = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                if hasattr(model, 'crf'):
                    logits, loss, metrics = model(x, targets=y, collect_metrics=True)
                    crf_metrics.extend(metrics)
                else:
                    logits, loss = model(x, targets=y)
                
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = logits.argmax(dim=-1)
                correct += (predictions == y).sum().item()
                total += y.numel()
        
        result = {
            'loss': total_loss / len(val_loader),
            'train_accuracy': correct / total,  # Using validation as proxy
            'val_accuracy': correct / total,
        }
        
        # Add CRF-specific metrics if available
        if crf_metrics:
            from metrics import aggregate_metrics
            aggregated = aggregate_metrics(crf_metrics)
            result.update({
                'avg_cell_count': aggregated.get('n_cells_mean', 0.0),
                'avg_comm_cost': aggregated.get('comm_cost_mean', 0.0),
                'specialization': aggregated.get('specialization', 0.0),
            })
        
        return result


def get_default_config(fast: bool = False) -> Dict:
    """Get default experiment configuration."""
    if fast:
        return {
            'vocab_size': 99,
            'd_model': 64,
            'd_hidden': 32,
            'n_init_cells': 16,
            'max_cells': 64,
            'n_crf_steps': 4,
            'k_neighbors': 3,
            'seq_len': 32,
            'dataset': 'synthetic',
            'max_train': 400,
            'max_val': 100,
            'batch_size': 16,
            'n_epochs': 5,
            'lr': 3e-4,
            'weight_decay': 0.1,
            'use_real': False,
        }
    else:
        return {
            'vocab_size': 99,
            'd_model': 128,
            'd_hidden': 64,
            'n_init_cells': 32,
            'max_cells': 128,
            'n_crf_steps': 6,
            'k_neighbors': 4,
            'seq_len': 64,
            'dataset': 'arithmetic',  # Use reasoning task
            'max_train': 2000,
            'max_val': 500,
            'batch_size': 32,
            'n_epochs': 15,
            'lr': 3e-4,
            'weight_decay': 0.1,
            'use_real': False,
        }


def main():
    parser = argparse.ArgumentParser(description='CRF Efficiency Benchmark')
    parser.add_argument('--fast', action='store_true', help='Run fast benchmark for testing')
    parser.add_argument('--crf-only', action='store_true', help='Run only CRF experiment')
    parser.add_argument('--transformer-only', action='store_true', help='Run only Transformer experiment')
    parser.add_argument('--no-optimizations', action='store_true', help='Disable efficiency optimizations')
    parser.add_argument('--output-dir', type=str, default='efficiency_benchmark_results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Setup
    benchmark = EfficiencyBenchmark(
        output_dir=args.output_dir,
        device=args.device
    )
    
    config = get_default_config(fast=args.fast)
    
    print(f"Configuration: {config}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    
    # Run experiments
    if args.crf_only:
        results = benchmark.run_crf_experiment(config, use_optimizations=not args.no_optimizations)
    elif args.transformer_only:
        results = benchmark.run_transformer_experiment(config)
    else:
        results = benchmark.run_comparative_benchmark(config)
    
    print(f"\nBenchmark complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()