"""Run the paper experiments in submission order.

This release entry point runs the section scripts in paper order and mirrors
figure/table outputs into ``src/spiking/results/``.

Usage:
  uv run python src/spiking/main.py
  uv run python src/spiking/main.py --stage simulation correctness
  uv run python src/spiking/main.py --scope main --skip-gpu
  uv run python src/spiking/main.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parent.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from spiking.config import BENCHMARKS, PAPER_RESULTS_DIR, PAPER_SRC_DIR, PROJECT_ROOT


STAGE_ORDER = (
    'shared_inputs',
    'simulation',
    'memorization',
    'correctness',
    'adjustment',
    'practical',
)

STAGE_LABELS = {
    'shared_inputs': 'Shared Inputs',
    'simulation': 'Section 3: Simulating Contamination',
    'memorization': 'Section 4.1: Memorization Predictors',
    'correctness': 'Section 4.2: Correctness Predictors',
    'adjustment': 'Section 4.3: Adjusting Estimates With Predictors',
    'practical': 'Section 5: Practical Considerations',
}

PLOT_SUFFIXES = {'.pdf', '.png', '.md', '.tex', '.html'}

# Release runs focus on the 8B / 500B-token Hubble pair.
RUN_EVAL_TASK_IDS = ('6', '7')  # standard, perturbed
PERTURBED_MODEL_INDEX = '3'     # 8b-500b within PERTURBED_MODELS
PAPER_MEM_ATTACKS = ('loss', 'zlib', 'min_k', 'min_k_plus_plus', 'reference')


@dataclass
class StepResult:
    stage: str
    name: str
    command: str
    gpu: bool
    skipped: bool
    success: bool
    duration_sec: float
    log_path: str | None


class PipelineRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logs_dir = PAPER_RESULTS_DIR / 'logs'
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        PAPER_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        self.results: list[StepResult] = []

    def run(self) -> None:
        for stage in self.args.stage:
            print(f'\n== {STAGE_LABELS[stage]} ==')
            try:
                getattr(self, f'_run_{stage}')()
            except Exception:
                if not self.args.continue_on_error:
                    raise

        self._write_manifest()

    def _run_script(
        self,
        *,
        stage: str,
        name: str,
        script_rel: str,
        script_args: list[str] | None = None,
        gpu: bool = False,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        script_args = script_args or []
        script_path = PROJECT_ROOT / script_rel
        cmd = [sys.executable, str(script_path), *script_args]
        command_str = shlex.join(cmd)

        if gpu and self.args.skip_gpu:
            print(f'  [SKIP gpu] {name}')
            self.results.append(StepResult(
                stage=stage,
                name=name,
                command=command_str,
                gpu=gpu,
                skipped=True,
                success=True,
                duration_sec=0.0,
                log_path=None,
            ))
            return

        if self.args.dry_run:
            print(f'  [DRY RUN] {command_str}')
            self.results.append(StepResult(
                stage=stage,
                name=name,
                command=command_str,
                gpu=gpu,
                skipped=True,
                success=True,
                duration_sec=0.0,
                log_path=None,
            ))
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_path = self.logs_dir / f'{timestamp}_{stage}_{name}.log'
        env = os.environ.copy()
        env['PYTHONPATH'] = self._pythonpath_with_src(env.get('PYTHONPATH'))
        if extra_env:
            env.update(extra_env)

        print(f'  [RUN] {name}')
        start = time.time()
        with log_path.open('w') as log_file:
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        duration = time.time() - start
        success = proc.returncode == 0

        self.results.append(StepResult(
            stage=stage,
            name=name,
            command=command_str,
            gpu=gpu,
            skipped=False,
            success=success,
            duration_sec=duration,
            log_path=str(log_path),
        ))

        if success:
            print(f'  [OK] {name} ({duration:.1f}s)')
            return

        tail = self._tail(log_path)
        print(f'  [FAIL] {name} ({duration:.1f}s)')
        if tail:
            print(tail)
        raise RuntimeError(f'Step failed: {name}')

    def _sync_stage_outputs(self, *relative_dirs: str) -> None:
        if self.args.dry_run:
            return

        for rel_dir in relative_dirs:
            src_dir = PAPER_SRC_DIR / rel_dir
            if not src_dir.exists():
                continue

            dest_dir = PAPER_RESULTS_DIR / rel_dir
            for path in src_dir.rglob('*'):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in PLOT_SUFFIXES:
                    continue
                target = dest_dir / path.relative_to(src_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    def _bootstrap_sample_efficiency_results(self) -> None:
        """Populate paper-local sample-efficiency inputs from the legacy cache."""
        dest_dir = PAPER_SRC_DIR / 'sample_efficiency' / 'results'
        required = [
            dest_dir / 'sample_efficiency_high.parquet',
            dest_dir / 'sample_efficiency_mid.parquet',
        ]
        if all(path.exists() for path in required):
            return

        legacy_dir = PROJECT_ROOT / 'experiments' / 'jw' / '55_sample_efficiency' / 'results'
        if not legacy_dir.exists():
            return

        dest_dir.mkdir(parents=True, exist_ok=True)
        for path in legacy_dir.glob('sample_efficiency*.parquet'):
            target = dest_dir / path.name
            if not target.exists():
                shutil.copy2(path, target)
                print(f'  [COPY] {path.relative_to(PROJECT_ROOT)} -> {target.relative_to(PROJECT_ROOT)}')

    def _pythonpath_with_src(self, existing: str | None) -> str:
        pieces = [str(SRC_ROOT)]
        if existing:
            pieces.append(existing)
        return os.pathsep.join(pieces)

    def _tail(self, path: Path, n_lines: int = 20) -> str:
        lines = path.read_text().splitlines()
        if not lines:
            return ''
        return '\n'.join(lines[-n_lines:])

    def _write_manifest(self) -> None:
        manifest_path = PAPER_RESULTS_DIR / 'run_manifest.json'
        payload = {
            'created_at': datetime.now().isoformat(),
            'scope': self.args.scope,
            'stages': self.args.stage,
            'results': [result.__dict__ for result in self.results],
        }
        manifest_path.write_text(json.dumps(payload, indent=2))
        print(f'\nWrote manifest: {manifest_path}')

    # Stages

    def _run_shared_inputs(self) -> None:
        self._run_script(
            stage='shared_inputs',
            name='eval_standard_8b_500b',
            script_rel='src/paper/data_generation/run_evals.py',
            gpu=True,
            extra_env={'SLURM_ARRAY_TASK_ID': RUN_EVAL_TASK_IDS[0]},
        )
        self._run_script(
            stage='shared_inputs',
            name='eval_perturbed_8b_500b',
            script_rel='src/paper/data_generation/run_evals.py',
            gpu=True,
            extra_env={'SLURM_ARRAY_TASK_ID': RUN_EVAL_TASK_IDS[1]},
        )

        for attack in PAPER_MEM_ATTACKS:
            self._run_script(
                stage='shared_inputs',
                name=f'mia_scores_{attack}',
                script_rel='src/paper/data_generation/run_mia_scores.py',
                script_args=['score', '--attack', attack, '--model', PERTURBED_MODEL_INDEX],
                gpu=True,
            )
        self._run_script(
            stage='shared_inputs',
            name='mia_scores_combine',
            script_rel='src/paper/data_generation/run_mia_scores.py',
            script_args=['combine', '--model-filter', '8b-500b'],
        )

        self._run_script(
            stage='shared_inputs',
            name='hidden_state_features',
            script_rel='src/paper/data_generation/run_hidden_states.py',
            script_args=['extract', '--model', PERTURBED_MODEL_INDEX],
            gpu=True,
        )

        self._run_script(
            stage='shared_inputs',
            name='llama_confidence_cache',
            script_rel='src/paper/data_generation/run_llm_confidence.py',
            script_args=['extract', '--external', 'llama'],
            gpu=True,
        )
        self._run_script(
            stage='shared_inputs',
            name='pythia_6_9b_confidence_cache',
            script_rel='src/paper/data_generation/run_llm_confidence.py',
            script_args=['extract', '--external', 'pythia', '--size', '6.9b'],
            gpu=True,
        )
        self._run_script(
            stage='shared_inputs',
            name='qwen_8b_confidence_cache',
            script_rel='src/paper/data_generation/run_llm_confidence.py',
            script_args=['extract', '--external', 'qwen', '--size', '8b'],
            gpu=True,
        )

        self._sync_stage_outputs('data_generation/figures')

    def _run_simulation(self) -> None:
        for benchmark in BENCHMARKS:
            self._run_script(
                stage='simulation',
                name=f'phase_diagram_{benchmark}',
                script_rel='src/paper/simulation/run.py',
                script_args=['--benchmark', benchmark],
            )

        if self.args.scope == 'full':
            self._run_script(
                stage='simulation',
                name='simulation_appendix',
                script_rel='src/paper/simulation/create_appendix.py',
            )

        self._sync_stage_outputs('simulation/figures')

    def _run_memorization(self) -> None:
        self._run_script(
            stage='memorization',
            name='memorization_probe_benchmark',
            script_rel='src/paper/memorization/run.py',
        )

        if self.args.scope == 'full':
            self._run_script(
                stage='memorization',
                name='memorization_probe_simulation',
                script_rel='src/paper/memorization/run_simulation.py',
            )

        self._sync_stage_outputs('memorization/figures')

    def _run_correctness(self) -> None:
        for benchmark in BENCHMARKS:
            self._run_script(
                stage='correctness',
                name=f'roberta_question_only_{benchmark}',
                script_rel='src/paper/correctness/run_roberta.py',
                script_args=[
                    '--benchmark', benchmark,
                    '--question-only',
                    '--lr', '5e-6',
                    '--freeze-layers', '0', '5',
                ],
                gpu=True,
            )

        self._run_script(
            stage='correctness',
            name='llama_correctness_probe',
            script_rel='src/paper/correctness/run_external_llm.py',
            script_args=['--external', 'llama'],
        )
        self._run_script(
            stage='correctness',
            name='pythia_6_9b_correctness_probe',
            script_rel='src/paper/correctness/run_external_llm.py',
            script_args=['--external', 'pythia', '--size', '6.9b'],
        )
        self._run_script(
            stage='correctness',
            name='qwen_8b_correctness_probe',
            script_rel='src/paper/correctness/run_external_llm.py',
            script_args=['--external', 'qwen', '--size', '8b'],
        )
        self._run_script(
            stage='correctness',
            name='correctness_probe_eval_table',
            script_rel='src/paper/correctness/run_evals.py',
            script_args=['--pythia-size', '6.9b', '--qwen-size', '8b', '--question-only'],
        )

        self._sync_stage_outputs('correctness/figures')

    def _run_adjustment(self) -> None:
        self._run_script(
            stage='adjustment',
            name='adjustment_simulation',
            script_rel='src/paper/adjustment/run_simulation.py',
        )

        if self.args.scope == 'full':
            suffix = 'all_8b-500b_min_k_plus_plus_standard_n500_g0.3_r1000'
            self._run_script(
                stage='adjustment',
                name='adjustment_calibration_plot',
                script_rel='src/paper/adjustment/plot_calibration.py',
                script_args=['--suffix', suffix],
            )

        self._sync_stage_outputs('adjustment/figures')

    def _run_practical(self) -> None:
        self._bootstrap_sample_efficiency_results()

        self._run_script(
            stage='practical',
            name='sample_efficiency_plot',
            script_rel='src/paper/sample_efficiency/plot.py',
        )

        self._run_script(
            stage='practical',
            name='memorization_transfer_simulation',
            script_rel='src/paper/probe_transfer/run_mem_sim.py',
        )

        visualize_args = ['--mode', 'mem', '--dose-group', 'all']
        if self.args.scope == 'main':
            visualize_args.extend(['--method', 'min_k'])
        self._run_script(
            stage='practical',
            name='memorization_transfer_visualization',
            script_rel='src/paper/probe_transfer/visualize.py',
            script_args=visualize_args,
        )

        self._sync_stage_outputs('sample_efficiency/figures', 'probe_transfer/figures')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run the paper experiments in submission order.')
    parser.add_argument(
        '--stage',
        nargs='+',
        default=list(STAGE_ORDER),
        choices=STAGE_ORDER,
        help='Subset of stages to run (default: all in paper order).',
    )
    parser.add_argument(
        '--scope',
        choices=['main', 'full'],
        default='full',
        help='Run only main-paper artifacts or include appendix-oriented stages too.',
    )
    parser.add_argument(
        '--skip-gpu',
        action='store_true',
        help='Skip GPU-requiring steps and run only CPU stages.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the planned commands without executing them.',
    )
    parser.add_argument(
        '--continue-on-error',
        action='store_true',
        help='Keep going after a failed step instead of stopping immediately.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = set(args.stage)
    args.stage = [stage for stage in STAGE_ORDER if stage in requested]
    runner = PipelineRunner(args)
    runner.run()


if __name__ == '__main__':
    main()
