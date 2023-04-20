"""
The evaluator class actually executes the evals.
"""
from evals.cli import oaieval
from evals.registry import Registry
from pathlib import Path
from typing import List, Optional, Tuple
import sys


class OAIRunArgs:
    def __init__(
        self,
        completion_fn: str,
        eval: str,
        extra_eval_params: str = "",
        max_samples: int = None,
        cache: bool = True,
        visible: bool = None,
        seed: int = 20220722,
        user: str = "",
        record_path: str = None,
        log_to_file: str = None,
        debug: bool = False,
        local_run: bool = True,
        dry_run: bool = False,
        dry_run_logging: bool = True,
    ):
        self.completion_fn = completion_fn
        self.eval = eval
        self.extra_eval_params = extra_eval_params
        self.max_samples = max_samples
        self.cache = cache
        self.visible = visible
        self.seed = seed
        self.user = user
        self.record_path = record_path
        self.log_to_file = log_to_file
        self.debug = debug
        self.local_run = local_run
        self.dry_run = dry_run
        self.dry_run_logging = dry_run_logging
        # create the record and logging paths if they don't exist
        Path(self.record_path).parent.mkdir(parents=True, exist_ok=True)
        # Path(self.log_to_file).parent.mkdir(parents=True, exist_ok=True)
        # Registry path should be the auto_gpt_benchmarking folder
        self.registry_path = None


class Evaluator:
    def __init__(self, oai_run_args: OAIRunArgs):
        self.oai_run_args = oai_run_args
        registry_path = Path(__file__).parent

        # add registry path to the python system path
        sys.path.append(str(registry_path))
        self.oai_run_args.registry_path = [registry_path]
        # self.registry = Registry([registry_path])

    def run(self):
        oaieval.run(self.oai_run_args)
