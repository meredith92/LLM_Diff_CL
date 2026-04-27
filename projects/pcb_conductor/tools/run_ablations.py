"""
Ablation Study Runner for PCB Conductor Continual Learning

Automatically generates ablation configs and runs experiments.
"""

import os
import json
import argparse
import subprocess
from pathlib import Path
from copy import deepcopy
from typing import Dict, List
import yaml


ABLATION_CONFIGS = {
    'full': {
        'description': 'Full method: Diffusion + Judge + LLM',
        'overrides': {
            'model.use_diffusion': True,
            'model.use_judge': True,
            'model.use_llm_judge': False,  # Set to False for reproducibility
        }
    },
    'no_diffusion': {
        'description': 'w/o Diffusion Prior',
        'overrides': {
            'model.use_diffusion': False,
            'model.use_judge': True,
            'model.use_llm_judge': False,
        }
    },
    'no_judge': {
        'description': 'w/o Structural Judge',
        'overrides': {
            'model.use_diffusion': True,
            'model.use_judge': False,
            'model.use_llm_judge': False,
        }
    },
    'diffusion_frozen': {
        'description': 'Diffusion Frozen (current config)',
        'overrides': {
            'model.use_diffusion': True,
            'model.use_judge': True,
            'model.use_llm_judge': False,
            'diffusion_frozen': True,  # Custom flag
        }
    },
    'diffusion_finetune': {
        'description': 'Diffusion Finetuned',
        'overrides': {
            'model.use_diffusion': True,
            'model.use_judge': True,
            'model.use_llm_judge': False,
            'diffusion_frozen': False,  # Custom flag
        }
    },
    'no_selective': {
        'description': 'w/o Selective Weight Map',
        'overrides': {
            'model.use_diffusion': True,
            'model.use_judge': True,
            'model.use_llm_judge': False,
            'model.use_selective': False,
        }
    },
    'no_skeleton': {
        'description': 'w/o Skeleton Consistency Loss',
        'overrides': {
            'model.use_diffusion': True,
            'model.use_judge': True,
            'model.use_llm_judge': False,
            'model.lam_skel': 0.0,  # Disable skeleton loss
        }
    },
}


class AblationStudyRunner:
    """Manage and run ablation experiments."""
    
    def __init__(self, 
                 base_config: str = 'projects/pcb_conductor/configs/segformer_mt_vb.py',
                 work_dir_base: str = 'work_dirs/ablation_study',
                 device: str = 'cuda',
                 launcher: str = 'none'):
        """
        Args:
            base_config: path to base config file
            work_dir_base: base directory for all ablation experiments
            device: 'cuda' or 'cpu'
            launcher: 'none', 'pytorch', 'slurm', 'mpi'
        """
        self.base_config = base_config
        self.work_dir_base = work_dir_base
        self.device = device
        self.launcher = launcher
        
        os.makedirs(self.work_dir_base, exist_ok=True)
        self.results = {}  # Store results of all runs
    
    def generate_config_overrides(self, ablation_name: str) -> List[str]:
        """
        Generate MMSeg command-line config override arguments.
        
        Args:
            ablation_name: key in ABLATION_CONFIGS
            
        Returns:
            List of strings like ['model.use_diffusion=False', ...]
        """
        if ablation_name not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation: {ablation_name}. Choose from {list(ABLATION_CONFIGS.keys())}")
        
        overrides = ABLATION_CONFIGS[ablation_name]['overrides']
        cfg_options = []
        
        for key, value in overrides.items():
            if isinstance(value, bool):
                value_str = str(value).lower()
            elif isinstance(value, (int, float)):
                value_str = str(value)
            else:
                value_str = f"'{value}'"
            
            cfg_options.append(f"{key}={value_str}")
        
        return cfg_options
    
    def run_single_ablation(self, ablation_name: str, resume: bool = False) -> Dict:
        """
        Run a single ablation experiment.
        
        Args:
            ablation_name: which ablation to run
            resume: whether to resume from latest checkpoint
            
        Returns:
            dict with experiment metadata and results
        """
        if ablation_name not in ABLATION_CONFIGS:
            raise ValueError(f"Unknown ablation: {ablation_name}")
        
        config_info = ABLATION_CONFIGS[ablation_name]
        work_dir = os.path.join(self.work_dir_base, ablation_name)
        os.makedirs(work_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"Running Ablation: {ablation_name}")
        print(f"Description: {config_info['description']}")
        print(f"Work Dir: {work_dir}")
        print(f"{'='*70}\n")
        
        # Build command
        cmd = [
            'python', 'tools/train.py',
            self.base_config,
            '--work-dir', work_dir,
            '--launcher', self.launcher,
            '--cfg-options',
        ]
        
        # Add config overrides
        cfg_options = self.generate_config_overrides(ablation_name)
        cmd.extend(cfg_options)
        
        # Add resume flag
        if resume:
            cmd.append('--resume')
        
        print(f"Command: {' '.join(cmd)}\n")
        
        # Run training
        try:
            result = subprocess.run(cmd, cwd=os.getcwd(), capture_output=False)
            
            if result.returncode == 0:
                print(f"\n✅ Ablation '{ablation_name}' completed successfully!")
                
                # Try to read metrics from work_dir
                metrics = self._extract_metrics(work_dir)
                
                return {
                    'ablation_name': ablation_name,
                    'description': config_info['description'],
                    'work_dir': work_dir,
                    'status': 'success',
                    'metrics': metrics,
                }
            else:
                print(f"\n❌ Ablation '{ablation_name}' failed with return code {result.returncode}")
                return {
                    'ablation_name': ablation_name,
                    'description': config_info['description'],
                    'work_dir': work_dir,
                    'status': 'failed',
                    'metrics': None,
                }
        
        except Exception as e:
            print(f"\n❌ Error running ablation '{ablation_name}': {str(e)}")
            return {
                'ablation_name': ablation_name,
                'description': config_info['description'],
                'work_dir': work_dir,
                'status': 'error',
                'metrics': None,
            }
    
    def _extract_metrics(self, work_dir: str) -> Dict:
        """
        Extract evaluation metrics from work directory.
        
        Looks for metrics.json or similar in the work_dir.
        """
        metrics_files = [
            os.path.join(work_dir, 'metrics.json'),
            os.path.join(work_dir, 'eval_results.json'),
        ]
        
        for mf in metrics_files:
            if os.path.exists(mf):
                try:
                    with open(mf, 'r') as f:
                        return json.load(f)
                except:
                    pass
        
        # Fallback: look in timestamp subdirectories
        for subdir in Path(work_dir).iterdir():
            if subdir.is_dir():
                for mf in metrics_files:
                    metrics_path = subdir / os.path.basename(mf)
                    if metrics_path.exists():
                        try:
                            with open(metrics_path, 'r') as f:
                                return json.load(f)
                        except:
                            pass
        
        print(f"⚠️  Could not find metrics in {work_dir}")
        return {}
    
    def run_all_ablations(self, 
                         ablation_names: List[str] = None,
                         resume: bool = False) -> Dict:
        """
        Run all ablation experiments in sequence.
        
        Args:
            ablation_names: list of ablation names to run. 
                           If None, run all in ABLATION_CONFIGS.
            resume: whether to resume from latest checkpoint for each
            
        Returns:
            dict mapping ablation_name -> result
        """
        if ablation_names is None:
            ablation_names = list(ABLATION_CONFIGS.keys())
        
        results = {}
        for ablation_name in ablation_names:
            result = self.run_single_ablation(ablation_name, resume=resume)
            results[ablation_name] = result
            self.results[ablation_name] = result
        
        return results
    
    def save_results(self, output_path: str = None):
        """Save ablation results to JSON."""
        if output_path is None:
            output_path = os.path.join(self.work_dir_base, 'ablation_results.json')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Saved ablation results to {output_path}")
    
    def print_summary_table(self):
        """Print summary table of all ablation results."""
        print("\n" + "="*80)
        print("ABLATION STUDY SUMMARY")
        print("="*80)
        
        # Header
        print(f"{'Ablation':<25} {'Description':<35} {'Status':<10}")
        print("-"*80)
        
        # Rows
        for ablation_name, result in self.results.items():
            desc = result['description'][:33]
            status = result['status'].upper()
            print(f"{ablation_name:<25} {desc:<35} {status:<10}")
        
        print("="*80 + "\n")
    
    def generate_comparison_table(self) -> str:
        """Generate comparison table for paper."""
        lines = []
        lines.append("| Method | mIoU | Δ Forgetting | Notes |")
        lines.append("|--------|------|-------------|-------|")
        
        for ablation_name in ['full', 'no_diffusion', 'no_judge', 'diffusion_frozen', 'diffusion_finetune']:
            if ablation_name in self.results:
                result = self.results[ablation_name]
                desc = result['description']
                status = result['status']
                
                # Extract mIoU if available
                metrics = result.get('metrics', {})
                miou = metrics.get('mIoU', 'N/A')
                
                lines.append(f"| {desc:<25} | {miou} | TBD | {status} |")
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--base-config', 
                       default='projects/pcb_conductor/configs/segformer_mt_vb.py',
                       help='Base config file path')
    parser.add_argument('--work-dir',
                       default='work_dirs/ablation_study',
                       help='Base work directory for ablations')
    parser.add_argument('--ablations',
                       nargs='+',
                       default=None,
                       help='Which ablations to run (default: all)')
    parser.add_argument('--resume',
                       action='store_true',
                       help='Resume from latest checkpoint')
    parser.add_argument('--launcher',
                       default='none',
                       choices=['none', 'pytorch', 'slurm', 'mpi'],
                       help='Job launcher type')
    
    args = parser.parse_args()
    
    # Create runner
    runner = AblationStudyRunner(
        base_config=args.base_config,
        work_dir_base=args.work_dir,
        launcher=args.launcher
    )
    
    # Run ablations
    if args.ablations is None:
        ablation_list = list(ABLATION_CONFIGS.keys())
    else:
        ablation_list = args.ablations
    
    results = runner.run_all_ablations(ablation_names=ablation_list, resume=args.resume)
    
    # Print and save results
    runner.print_summary_table()
    runner.save_results()
    
    print("\n📊 Comparison Table (for paper):")
    print(runner.generate_comparison_table())
    
    return results


if __name__ == '__main__':
    main()
