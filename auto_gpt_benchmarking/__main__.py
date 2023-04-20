"""
This is the main evaluation file. In it you can specify the following:

1. The number of threads to use for evaluation. This is set to 1 by default.And will remain that way until we can spin
 up containers on command
2. The timeout for each thread. This is set to 60 seconds by default. This is the amount of time each thread will run
 for before it is killed when evaluating an agent
3. The path to the AutoGPT code. This is a required parameter as we do not know where your code lives.
4. The evals you would like to run. The options here are any OpenAI eval, or any of the evals defined in this repository


What this file does is it parses the params given and then runs the evals with OpenAI's evals framework.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import yaml
from datetime import datetime



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "eval", type=str, help="Name of an eval. See registry.")
    parser.add_argument(
        "--completion-fn",
        type=str,
        dest="completion_fn",
        default="auto_gpt_completion_fn",
        help="One or more CompletionFn URLs, separated by commas (,). "
             "A CompletionFn can either be the name of a model available in the OpenAI API or a key in the registry "
             "(see evals/registry/completion_fns).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="The timeout for each thread",
    )
    parser.add_argument(
        "--auto-gpt-path",
        type=str,
        default=None,
        help="The path to the AutoGPT code. This updates auto_gpt_competion_fn.yaml in completion fns. "
             "So you only need to set this once.",
    )
    parser.add_argument("--extra_eval_params", type=str, default="")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--visible", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--seed", type=int, default=20220722)
    parser.add_argument("--user", type=str, default="")
    parser.add_argument("--record_path", type=str, default=str(Path(
        __file__).parent.parent / "data" / f"eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"))
    parser.add_argument(
        "--log_to_file", type=str, default=None,  # default=str(
        #   Path(__file__).parent.parent / "data" / "log" / "log.txt"
        #  ), help="Log to a file instead of stdout"
    )
    parser.add_argument(
        "--debug", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--local-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--dry-run-logging",
                        action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def update_yaml_with_auto_gpt_path(yaml_path: str, auto_gpt_path: str or None) -> Path:
    """
    If there is a given auto_gpt_path, then we need to update the yaml file to include it in the system path
    If we don't have one. Then we get the path from the yaml.
    If none exists in the yaml and we don't have a path then we raise an exception.
    :param yaml_path: The path to the yaml file
    :param auto_gpt_path: The path to the AutoGPT code
    :return: The path to the AutoGPT code
    """
    with open(yaml_path, "r") as f:
        yaml_data = yaml.safe_load(f)
    if yaml_data["auto_gpt_completion_fn"]["args"]["auto_gpt_path"] is None and auto_gpt_path is None:
        raise Exception(
            "You must specify a auto_gpt_path in the yaml file or pass it in as a parameter")
    if auto_gpt_path is None:
        auto_gpt_path = yaml_data["auto_gpt_completion_fn"]["args"]["auto_gpt_path"]
    if auto_gpt_path is not None:
        yaml_data["auto_gpt_completion_fn"]["args"]["auto_gpt_path"] = auto_gpt_path
    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f)

    return Path(auto_gpt_path).absolute()


def load_env_file(env_path: Path):
    if not env_path.exists():
        raise FileNotFoundError('You must set the OpenAI key in the AutoGPT env file. '
                                'We need your api keys to start the AutoGPT agent and use OpenAI evals')
    with open(env_path, "r") as f:
        # find the OPENAI_API_KEY key split it from the equals sign and assign it so OpenAI evals can use it.
        for line in f.readlines():
            if line.startswith("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = line.split("=")[1].strip()
                break


if __name__ == "__main__":
    args = parse_args()
    # do not run in multiprocessing mode We do not use this right now, as it disables OpenAI's timeouts :(
    # os.environ["EVALS_SEQUENTIAL"] = "1"
    os.environ["EVALS_THREAD_TIMEOUT"] = str(args.timeout)
    os.environ["EVALS_THREADS"] = str(1)

    # Update the yaml file with the auto_gpt_path
    autogpt_path = update_yaml_with_auto_gpt_path(
        str(Path(__file__).parent / "completion_fns" /
            "auto_gpt_completion_fn.yaml"),
        args.auto_gpt_path
    )

    # Add the benchmarks path to the system path so we can import auto_gpt_benchmarking
    sys.path.append(str(Path(__file__).parent.parent.absolute()))

    # load all of the environment variables in the auto-gpt path/.env file
    load_env_file(Path(autogpt_path) / ".env")

    # Obviously, a top level import would be better. This allows us to set the API key with the env file, as it gets
    # set in the evaluator. We can't set it before the import because the import will fail without an API key.
    from auto_gpt_benchmarking.Evaluator import Evaluator, OAIRunArgs
    run_args = OAIRunArgs(
        completion_fn=args.completion_fn,
        eval=args.eval,
        extra_eval_params=args.extra_eval_params,
        max_samples=args.max_samples,
        cache=args.cache,
        visible=args.visible,
        seed=args.seed,
        user=args.user,
        record_path=args.record_path,
        log_to_file=args.log_to_file,
        debug=args.debug,
        local_run=args.local_run,
        dry_run=args.dry_run,
        dry_run_logging=args.dry_run_logging)

    # Run the evals
    evaluator = Evaluator(
        run_args
    )
    evaluator.run()
