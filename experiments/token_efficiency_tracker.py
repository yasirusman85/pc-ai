"""
token_efficiency_tracker.py — Track and analyze token efficiency for CRF vs Transformer
================================================================================
Tracks metrics to validate the hypothesis:
  "CRF achieves Transformer-level reasoning with 10-100× fewer training tokens"

Key metrics:
  - Accuracy vs training tokens
  - Accuracy vs GPU-hours  
  - Accuracy vs FLOPs
  - Accuracy vs wall-clock training time
  - Sample efficiency curves
  - Computational efficiency ratios
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import torch
import numpy as np


@dataclass
class TrainingSnapshot:
    """Snapshot of training state at a point in time."""
    epoch: int
    training_tokens: int
    wall_time_hours: float
    gpu_hours: float
    total_flops: int
    
    # Performance metrics
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    reasoning_score: float
    
    # CRF-specific metrics
    avg_cell_count: float
    avg_comm_cost: float
    specialization: float
    
    # Efficiency metrics
    tokens_per_accuracy_point: float
    flops_per_accuracy_point: float
    hours_per_accuracy_point: float


class TokenEfficiencyTracker:
    """
    Tracks training efficiency metrics to validate the core hypothesis.
    
    Hypothesis: CRF achieves Transformer-level reasoning with 10-100× fewer training tokens
    
    This tracker monitors:
    1. Sample efficiency: How quickly accuracy improves with training tokens
    2. Compute efficiency: Accuracy per FLOP and per GPU-hour
    3. Time efficiency: Accuracy per wall-clock hour
    4. Reasoning efficiency: Quality of reasoning per unit compute
    """
    
    def __init__(self, model_name: str, output_dir: str = "efficiency_results"):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Training state
        self.snapshots: List[TrainingSnapshot] = []
        self.start_time = time.time()
        self.start_gpu_time = self._get_gpu_time()
        
        # Token counting
        self.training_tokens = 0
        self.epoch_tokens = 0
        
        # FLOP tracking
        self.total_flops = 0
        self.epoch_flops = 0
        
        # Performance baselines
        self.target_accuracy = 0.90  # Target: 90% of Transformer performance
        self.transformer_baseline = None
        
    def _get_gpu_time(self) -> float:
        """Get GPU time in hours."""
        try:
            if torch.cuda.is_available():
                return torch.cuda.cuda_timer() / 3600.0  # Convert to hours
        except:
            pass
        return 0.0
    
    def record_batch(self, batch_size: int, seq_len: int, estimated_flops: int):
        """
        Record statistics for a training batch.
        
        Args:
            batch_size: Number of sequences in batch
            seq_len: Length of each sequence
            estimated_flops: Estimated FLOPs for this batch
        """
        # Count training tokens
        batch_tokens = batch_size * seq_len
        self.training_tokens += batch_tokens
        self.epoch_tokens += batch_tokens
        
        # Track FLOPs
        self.total_flops += estimated_flops
        self.epoch_flops += estimated_flops
    
    def record_epoch(self, epoch: int, metrics: Dict):
        """
        Record end-of-epoch metrics.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary containing performance metrics
        """
        current_time = time.time()
        wall_time_hours = (current_time - self.start_time) / 3600.0
        gpu_hours = self._get_gpu_time() - self.start_gpu_time
        
        # Calculate efficiency metrics
        if metrics.get('val_accuracy', 0) > 0:
            tokens_per_acc = self.training_tokens / metrics['val_accuracy']
            flops_per_acc = self.total_flops / metrics['val_accuracy']
            hours_per_acc = wall_time_hours / metrics['val_accuracy']
        else:
            tokens_per_acc = float('inf')
            flops_per_acc = float('inf')
            hours_per_acc = float('inf')
        
        snapshot = TrainingSnapshot(
            epoch=epoch,
            training_tokens=self.training_tokens,
            wall_time_hours=wall_time_hours,
            gpu_hours=gpu_hours,
            total_flops=self.total_flops,
            train_loss=metrics.get('train_loss', 0.0),
            val_loss=metrics.get('val_loss', 0.0),
            train_accuracy=metrics.get('train_accuracy', 0.0),
            val_accuracy=metrics.get('val_accuracy', 0.0),
            reasoning_score=metrics.get('reasoning_score', 0.0),
            avg_cell_count=metrics.get('avg_cell_count', 0.0),
            avg_comm_cost=metrics.get('avg_comm_cost', 0.0),
            specialization=metrics.get('specialization', 0.0),
            tokens_per_accuracy_point=tokens_per_acc,
            flops_per_accuracy_point=flops_per_acc,
            hours_per_accuracy_point=hours_per_acc,
        )
        
        self.snapshots.append(snapshot)
        
        # Reset epoch counters
        self.epoch_tokens = 0
        self.epoch_flops = 0
        
        # Save intermediate results
        self.save_snapshot(snapshot)
    
    def set_transformer_baseline(self, baseline_metrics: Dict):
        """
        Set Transformer baseline for comparison.
        
        Args:
            baseline_metrics: Dictionary with Transformer efficiency metrics
        """
        self.transformer_baseline = baseline_metrics
    
    def save_snapshot(self, snapshot: TrainingSnapshot):
        """Save individual snapshot to file."""
        snapshot_file = self.output_dir / f"snapshot_epoch_{snapshot.epoch}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(asdict(snapshot), f, indent=2)
    
    def save_all_snapshots(self):
        """Save all snapshots to a single file."""
        output_file = self.output_dir / f"{self.model_name}_efficiency.json"
        with open(output_file, 'w') as f:
            json.dump([asdict(s) for s in self.snapshots], f, indent=2)
    
    def analyze_efficiency(self) -> Dict:
        """
        Analyze efficiency metrics and compare to baseline.
        
        Returns:
            Dictionary with efficiency analysis results
        """
        if not self.snapshots:
            return {"error": "No snapshots available"}
        
        latest = self.snapshots[-1]
        
        analysis = {
            "model_name": self.model_name,
            "final_metrics": asdict(latest),
            "efficiency_analysis": {},
            "hypothesis_validation": {},
        }
        
        # Calculate key efficiency ratios
        if self.transformer_baseline:
            analysis["efficiency_analysis"]["vs_transformer"] = {
                "token_ratio": latest.training_tokens / self.transformer_baseline.get('training_tokens', 1),
                "flop_ratio": latest.total_flops / self.transformer_baseline.get('total_flops', 1),
                "time_ratio": latest.wall_time_hours / self.transformer_baseline.get('wall_time_hours', 1),
                "accuracy_ratio": latest.val_accuracy / self.transformer_baseline.get('val_accuracy', 1),
            }
        
        # Sample efficiency analysis
        analysis["efficiency_analysis"]["sample_efficiency"] = self._analyze_sample_efficiency()
        
        # Compute efficiency analysis
        analysis["efficiency_analysis"]["compute_efficiency"] = self._analyze_compute_efficiency()
        
        # Time efficiency analysis
        analysis["efficiency_analysis"]["time_efficiency"] = self._analyze_time_efficiency()
        
        # Hypothesis validation
        analysis["hypothesis_validation"] = self._validate_hypothesis()
        
        return analysis
    
    def _analyze_sample_efficiency(self) -> Dict:
        """Analyze sample efficiency (accuracy vs tokens)."""
        if not self.snapshots:
            return {}
        
        # Find tokens needed to reach various accuracy thresholds
        thresholds = [0.5, 0.7, 0.8, 0.9]
        tokens_to_threshold = {}
        
        for threshold in thresholds:
            for snapshot in self.snapshots:
                if snapshot.val_accuracy >= threshold:
                    tokens_to_threshold[f"{int(threshold*100)}%"] = snapshot.training_tokens
                    break
        
        # Calculate area under accuracy-tokens curve
        tokens = [s.training_tokens for s in self.snapshots]
        accuracies = [s.val_accuracy for s in self.snapshots]
        auc = np.trapz(accuracies, tokens) if len(tokens) > 1 else 0.0
        
        return {
            "tokens_to_threshold": tokens_to_threshold,
            "area_under_curve": float(auc),
            "final_tokens_per_accuracy": float(self.snapshots[-1].tokens_per_accuracy_point),
        }
    
    def _analyze_compute_efficiency(self) -> Dict:
        """Analyze compute efficiency (accuracy vs FLOPs)."""
        if not self.snapshots:
            return {}
        
        # Find FLOPs needed to reach accuracy thresholds
        thresholds = [0.5, 0.7, 0.8, 0.9]
        flops_to_threshold = {}
        
        for threshold in thresholds:
            for snapshot in self.snapshots:
                if snapshot.val_accuracy >= threshold:
                    flops_to_threshold[f"{int(threshold*100)}%"] = snapshot.total_flops
                    break
        
        return {
            "flops_to_threshold": flops_to_threshold,
            "final_flops_per_accuracy": float(self.snapshots[-1].flops_per_accuracy_point),
        }
    
    def _analyze_time_efficiency(self) -> Dict:
        """Analyze time efficiency (accuracy vs wall-clock time)."""
        if not self.snapshots:
            return {}
        
        # Find time needed to reach accuracy thresholds
        thresholds = [0.5, 0.7, 0.8, 0.9]
        time_to_threshold = {}
        
        for threshold in thresholds:
            for snapshot in self.snapshots:
                if snapshot.val_accuracy >= threshold:
                    time_to_threshold[f"{int(threshold*100)}%"] = snapshot.wall_time_hours
                    break
        
        return {
            "hours_to_threshold": time_to_threshold,
            "final_hours_per_accuracy": float(self.snapshots[-1].hours_per_accuracy_point),
        }
    
    def _validate_hypothesis(self) -> Dict:
        """
        Validate the core hypothesis.
        
        Hypothesis: CRF achieves Transformer-level reasoning with 10-100× fewer training tokens
        """
        if not self.snapshots or not self.transformer_baseline:
            return {"status": "insufficient_data"}
        
        latest = self.snapshots[-1]
        baseline = self.transformer_baseline
        
        # Calculate efficiency ratios
        token_ratio = latest.training_tokens / baseline.get('training_tokens', 1)
        flop_ratio = latest.total_flops / baseline.get('total_flops', 1)
        time_ratio = latest.wall_time_hours / baseline.get('wall_time_hours', 1)
        accuracy_ratio = latest.val_accuracy / baseline.get('val_accuracy', 1)
        
        # Check hypothesis conditions
        hypothesis_valid = (
            token_ratio <= 0.1 and  # 10× fewer tokens
            accuracy_ratio >= 0.9    # 90% of Transformer accuracy
        )
        
        return {
            "status": "validated" if hypothesis_valid else "not_validated",
            "token_ratio": float(token_ratio),
            "flop_ratio": float(flop_ratio),
            "time_ratio": float(time_ratio),
            "accuracy_ratio": float(accuracy_ratio),
            "hypothesis_met": hypothesis_valid,
            "efficiency_gain": {
                "token": f"{1/token_ratio:.1f}×" if token_ratio > 0 else "N/A",
                "flop": f"{1/flop_ratio:.1f}×" if flop_ratio > 0 else "N/A",
                "time": f"{1/time_ratio:.1f}×" if time_ratio > 0 else "N/A",
            }
        }
    
    def generate_efficiency_report(self) -> str:
        """Generate human-readable efficiency report."""
        analysis = self.analyze_efficiency()
        
        report = []
        report.append("=" * 80)
        report.append(f"TOKEN EFFICIENCY REPORT: {self.model_name}")
        report.append("=" * 80)
        
        if "error" in analysis:
            report.append(f"Error: {analysis['error']}")
            return "\n".join(report)
        
        latest = analysis["final_metrics"]
        report.append(f"\nFinal Performance:")
        report.append(f"  Training Tokens: {latest['training_tokens']:,}")
        report.append(f"  Wall Time: {latest['wall_time_hours']:.2f} hours")
        report.append(f"  Total FLOPs: {latest['total_flops']:,}")
        report.append(f"  Validation Accuracy: {latest['val_accuracy']:.3f}")
        report.append(f"  Reasoning Score: {latest['reasoning_score']:.3f}")
        
        report.append(f"\nEfficiency Metrics:")
        report.append(f"  Tokens per Accuracy Point: {latest['tokens_per_accuracy_point']:.0f}")
        report.append(f"  FLOPs per Accuracy Point: {latest['flops_per_accuracy_point']:.0f}")
        report.append(f"  Hours per Accuracy Point: {latest['hours_per_accuracy_point']:.3f}")
        
        if "vs_transformer" in analysis["efficiency_analysis"]:
            vs_transformer = analysis["efficiency_analysis"]["vs_transformer"]
            report.append(f"\nComparison vs Transformer:")
            report.append(f"  Token Ratio: {vs_transformer['token_ratio']:.3f}×")
            report.append(f"  FLOP Ratio: {vs_transformer['flop_ratio']:.3f}×")
            report.append(f"  Time Ratio: {vs_transformer['time_ratio']:.3f}×")
            report.append(f"  Accuracy Ratio: {vs_transformer['accuracy_ratio']:.3f}×")
        
        if "sample_efficiency" in analysis["efficiency_analysis"]:
            sample_eff = analysis["efficiency_analysis"]["sample_efficiency"]
            report.append(f"\nSample Efficiency:")
            report.append(f"  Tokens to 50%: {sample_eff['tokens_to_threshold'].get('50%', 'N/A'):,}")
            report.append(f"  Tokens to 80%: {sample_eff['tokens_to_threshold'].get('80%', 'N/A'):,}")
            report.append(f"  AUC: {sample_eff['area_under_curve']:.0f}")
        
        hypothesis = analysis["hypothesis_validation"]
        if hypothesis.get("status") == "validated":
            report.append(f"\nHYPOTHESIS VALIDATION: {'✓ PASSED' if hypothesis['hypothesis_met'] else '✗ FAILED'}")
            report.append(f"  Token Efficiency: {hypothesis['efficiency_gain']['token']}")
            report.append(f"  FLOP Efficiency: {hypothesis['efficiency_gain']['flop']}")
            report.append(f"  Time Efficiency: {hypothesis['efficiency_gain']['time']}")
        else:
            report.append(f"\nHYPOTHESIS VALIDATION: {hypothesis['status'].upper()}")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def plot_efficiency_curves(self, save_path: Optional[str] = None):
        """
        Plot efficiency curves (accuracy vs tokens, FLOPs, time).
        
        Args:
            save_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib not available, skipping plots")
            return
        
        if not self.snapshots:
            print("No data to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract data
        tokens = [s.training_tokens for s in self.snapshots]
        flops = [s.total_flops for s in self.snapshots]
        hours = [s.wall_time_hours for s in self.snapshots]
        accuracies = [s.val_accuracy for s in self.snapshots]
        
        # Plot 1: Accuracy vs Training Tokens
        axes[0].plot(tokens, accuracies, 'b-', linewidth=2, marker='o')
        axes[0].set_xlabel('Training Tokens')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Sample Efficiency')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Accuracy vs FLOPs
        axes[1].plot(flops, accuracies, 'r-', linewidth=2, marker='s')
        axes[1].set_xlabel('Total FLOPs')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Compute Efficiency')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Accuracy vs Wall-clock Time
        axes[2].plot(hours, accuracies, 'g-', linewidth=2, marker='^')
        axes[2].set_xlabel('Wall-clock Time (hours)')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Time Efficiency')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=120, bbox_inches='tight')
            print(f"Saved efficiency plot to {save_path}")
        else:
            plt.show()
        
        plt.close()


class ComparativeEfficiencyAnalyzer:
    """
    Compare efficiency between CRF and Transformer baselines.
    
    Runs comparative experiments to validate the core hypothesis.
    """
    
    def __init__(self, output_dir: str = "comparative_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.crf_tracker = None
        self.transformer_tracker = None
        
    def setup_crf_tracking(self, model_name: str = "CRF"):
        """Setup tracking for CRF model."""
        self.crf_tracker = TokenEfficiencyTracker(
            model_name=model_name,
            output_dir=str(self.output_dir / "crf")
        )
        return self.crf_tracker
    
    def setup_transformer_tracking(self, model_name: str = "Transformer"):
        """Setup tracking for Transformer baseline."""
        self.transformer_tracker = TokenEfficiencyTracker(
            model_name=model_name,
            output_dir=str(self.output_dir / "transformer")
        )
        return self.transformer_tracker
    
    def run_comparative_analysis(self) -> Dict:
        """
        Run comparative efficiency analysis.
        
        Returns:
            Dictionary with comparative results
        """
        if not self.crf_tracker or not self.transformer_tracker:
            return {"error": "Both trackers must be setup"}
        
        crf_analysis = self.crf_tracker.analyze_efficiency()
        transformer_analysis = self.transformer_tracker.analyze_efficiency()
        
        comparative = {
            "crf_final": crf_analysis.get("final_metrics", {}),
            "transformer_final": transformer_analysis.get("final_metrics", {}),
            "comparison": {},
            "hypothesis_result": {},
        }
        
        # Calculate comparative ratios
        if crf_analysis.get("final_metrics") and transformer_analysis.get("final_metrics"):
            crf_final = crf_analysis["final_metrics"]
            transformer_final = transformer_analysis["final_metrics"]
            
            comparative["comparison"] = {
                "token_ratio": crf_final["training_tokens"] / max(1, transformer_final["training_tokens"]),
                "flop_ratio": crf_final["total_flops"] / max(1, transformer_final["total_flops"]),
                "time_ratio": crf_final["wall_time_hours"] / max(1, transformer_final["wall_time_hours"]),
                "accuracy_ratio": crf_final["val_accuracy"] / max(1, transformer_final["val_accuracy"]),
            }
            
            # Validate hypothesis
            comp = comparative["comparison"]
            hypothesis_met = (
                comp["token_ratio"] <= 0.1 and  # 10× fewer tokens
                comp["accuracy_ratio"] >= 0.9    # 90% of Transformer accuracy
            )
            
            comparative["hypothesis_result"] = {
                "met": hypothesis_met,
                "token_efficiency_gain": f"{1/comp['token_ratio']:.1f}×",
                "flop_efficiency_gain": f"{1/comp['flop_ratio']:.1f}×",
                "time_efficiency_gain": f"{1/comp['time_ratio']:.1f}×",
            }
        
        # Save comparative results
        output_file = self.output_dir / "comparative_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(comparative, f, indent=2)
        
        return comparative
    
    def generate_comparative_report(self) -> str:
        """Generate comparative efficiency report."""
        comparative = self.run_comparative_analysis()
        
        report = []
        report.append("=" * 80)
        report.append("COMPARATIVE EFFICIENCY ANALYSIS: CRF vs Transformer")
        report.append("=" * 80)
        
        if "error" in comparative:
            report.append(f"Error: {comparative['error']}")
            return "\n".join(report)
        
        comp = comparative.get("comparison", {})
        hypothesis = comparative.get("hypothesis_result", {})
        
        report.append(f"\nFinal Performance Comparison:")
        report.append(f"  CRF Tokens: {comparative['crf_final'].get('training_tokens', 0):,}")
        report.append(f"  Transformer Tokens: {comparative['transformer_final'].get('training_tokens', 0):,}")
        report.append(f"  Token Ratio: {comp.get('token_ratio', 0):.3f}×")
        
        report.append(f"\n  CRF FLOPs: {comparative['crf_final'].get('total_flops', 0):,}")
        report.append(f"  Transformer FLOPs: {comparative['transformer_final'].get('total_flops', 0):,}")
        report.append(f"  FLOP Ratio: {comp.get('flop_ratio', 0):.3f}×")
        
        report.append(f"\n  CRF Time: {comparative['crf_final'].get('wall_time_hours', 0):.2f}h")
        report.append(f"  Transformer Time: {comparative['transformer_final'].get('wall_time_hours', 0):.2f}h")
        report.append(f"  Time Ratio: {comp.get('time_ratio', 0):.3f}×")
        
        report.append(f"\n  CRF Accuracy: {comparative['crf_final'].get('val_accuracy', 0):.3f}")
        report.append(f"  Transformer Accuracy: {comparative['transformer_final'].get('val_accuracy', 0):.3f}")
        report.append(f"  Accuracy Ratio: {comp.get('accuracy_ratio', 0):.3f}×")
        
        if hypothesis:
            report.append(f"\nHYPOTHESIS VALIDATION:")
            report.append(f"  Status: {'✓ PASSED' if hypothesis['met'] else '✗ FAILED'}")
            report.append(f"  Token Efficiency Gain: {hypothesis['token_efficiency_gain']}")
            report.append(f"  FLOP Efficiency Gain: {hypothesis['flop_efficiency_gain']}")
            report.append(f"  Time Efficiency Gain: {hypothesis['time_efficiency_gain']}")
        
        report.append("=" * 80)
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test the token efficiency tracker
    print("Testing Token Efficiency Tracker...")
    
    tracker = TokenEfficiencyTracker("test_model")
    
    # Simulate training
    for epoch in range(5):
        # Simulate batch processing
        for _ in range(10):
            tracker.record_batch(batch_size=32, seq_len=64, estimated_flops=1000000)
        
        # Simulate metrics
        metrics = {
            'train_loss': 2.0 - epoch * 0.3,
            'val_loss': 2.2 - epoch * 0.25,
            'train_accuracy': 0.3 + epoch * 0.12,
            'val_accuracy': 0.25 + epoch * 0.10,
            'reasoning_score': 0.2 + epoch * 0.08,
            'avg_cell_count': 32 + epoch * 2,
            'avg_comm_cost': 1000 + epoch * 50,
            'specialization': 0.5 + epoch * 0.02,
        }
        
        tracker.record_epoch(epoch, metrics)
    
    # Generate report
    report = tracker.generate_efficiency_report()
    print(report)
    
    # Save results
    tracker.save_all_snapshots()
    analysis = tracker.analyze_efficiency()
    print(f"\nAnalysis saved to efficiency_results/")
    
    print("\nToken Efficiency Tracker tested successfully!")