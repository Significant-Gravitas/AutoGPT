# import pydevd_pycharm

# pydevd_pycharm.settrace(
#     "localhost", port=9739, stdoutToServer=True, stderrToServer=True
# )
from .utils.data_types import AgentBenchmarkConfig
import sys
import json
from .reports.ReportManager import ReportManager

def get_agent_benchmark_config() -> AgentBenchmarkConfig:
    if "--agent-config" in sys.argv:
        agent_benchmark_config_path = sys.argv[sys.argv.index("--agent-config") + 1]
    else:
        print(sys.argv)
    try:
        with open(agent_benchmark_config_path, "r") as f:
            agent_benchmark_config = AgentBenchmarkConfig(**json.load(f))
            agent_benchmark_config.agent_benchmark_config_path = (
                agent_benchmark_config_path
            )
            return agent_benchmark_config
    except json.JSONDecodeError:
        print("Error: benchmark_config.json is not a valid JSON file.")
        raise


def get_report_managers() -> tuple[ReportManager, ReportManager, ReportManager]:
    agent_benchmark_config = get_agent_benchmark_config()
    # tests that consistently pass are considered regression tests
    REGRESSION_MANAGER = ReportManager(
        agent_benchmark_config.get_regression_reports_path()
    )

    # print(f"Using {REPORTS_PATH} for reports")
    # user facing reporting information
    INFO_MANAGER = ReportManager(
        str(agent_benchmark_config.get_reports_path() / "report.json")
    )

    # internal db step in replacement track pass/fail rate
    INTERNAL_INFO_MANAGER = ReportManager(
        agent_benchmark_config.get_success_rate_path()
    )

    return REGRESSION_MANAGER, INFO_MANAGER, INTERNAL_INFO_MANAGER



(REGRESSION_MANAGER, INFO_MANAGER, INTERNAL_INFO_MANAGER) = get_report_managers()
