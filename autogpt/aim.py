import copy
import json
from typing import Any, Optional


def import_aim() -> Any:
    try:
        import aim
    except ImportError:
        raise ImportError(
            "To use the Aim callback manager you need to have the"
            " `aim` python package installed."
            "Please install it with `pip install aim`"
        )
    return aim


class AimCallback:
    """
    AimCallback callback function.

    Args:
        repo (:obj:`str`, optional): Aim repository path or Repo object to which Run object is bound.
            If skipped, default Repo is used.
        experiment_name (:obj:`str`, optional): Sets Run's `experiment` property. 'default' if not specified.
            Can be used later to query runs/sequences.
        system_tracking_interval (:obj:`int`, optional): Sets the tracking interval in seconds for system usage
            metrics (CPU, Memory, etc.). Set to `None` to disable system metrics tracking.
        log_system_params (:obj:`bool`, optional): Enable/Disable logging of system params such as installed packages,
            git info, environment variables, etc.
        capture_terminal_logs (:obj:`bool`, optional): Enable/Disable logging of terminal input/outputs.
        log_keys: (:obj:`bool`, optional) Triggers key(e.g. openai_api_key) tracking.
    """

    def __init__(
        self,
        repo: Optional[str] = None,
        experiment_name: Optional[str] = None,
        system_tracking_interval: Optional[int] = 10,
        log_system_params: Optional[bool] = True,
        capture_terminal_logs: Optional[bool] = True,
        log_keys: Optional[bool] = False,
    ):
        self.repo = repo
        self.experiment_name = experiment_name
        self.system_tracking_interval = system_tracking_interval
        self.log_system_params = log_system_params
        self.capture_terminal_logs = capture_terminal_logs
        self.log_keys = log_keys
        self._run = None
        self._run_hash = None

    def track(self, logs, context, step=None):
        for k, v in logs.items():
            if isinstance(v, list):
                if len(v) == 1:
                    v = v[0]
                else:
                    raise NotImplementedError(f"number of items in {k} are more than 1")
            self._run.track(v, k, step=step, context=context, epoch=self.epoch)

    def track_text(self, text, name, context=None):
        aim = import_aim()
        self._run.track(aim.Text(text), name=name, context=context)

    @property
    def experiment(self):
        if not self._run:
            self.setup()
        return self._run

    def setup(self, args=None):
        aim = import_aim()
        if not self._run:
            if self._run_hash:
                self._run = aim.Run(
                    self._run_hash,
                    repo=self.repo,
                    system_tracking_interval=self.system_tracking_interval,
                    log_system_params=self.log_system_params,
                    capture_terminal_logs=self.capture_terminal_logs,
                )
            else:
                self._run = aim.Run(
                    repo=self.repo,
                    experiment=self.experiment_name,
                    system_tracking_interval=self.system_tracking_interval,
                    log_system_params=self.log_system_params,
                    capture_terminal_logs=self.capture_terminal_logs,
                )
                self._run_hash = self._run.hash

        # Log config parameters
        if args:
            for key, arg in args.items():
                arg = copy.copy(arg)
                try:
                    self._run.set(key, arg)
                except TypeError:
                    self._run.set(key, self._set(arg))

    def _set(self, args):
        args = args.__dict__
        keys = list(args.keys())
        for key in keys:
            if not self.log_keys and key.endswith("_key"):
                args.pop(key)
                continue
            try:
                json.dumps(args[key])
            except TypeError:
                args.pop(key)
        return args

    def __del__(self):
        if self._run and self._run.active:
            self._run.close()
