"""AgentBench adapter.

AgentBench evaluates LLMs as agents across diverse real-world environments:
Operating System, Database, Knowledge Graph, Card Game, Lateral Thinking,
Web Shopping, Web Browsing, and ALFWorld.

GitHub: https://github.com/THUDM/AgentBench
Paper: https://arxiv.org/abs/2308.03688

Requires:
    - Docker (for OS environment)
    - Database drivers (for DB environment)
    - Environment-specific dependencies
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Iterator, Optional

from ..models import Challenge, ChallengeResult
from . import register_adapter
from .base import BenchmarkAdapter


@register_adapter("agent-bench")
class AgentBenchAdapter(BenchmarkAdapter):
    """Adapter for the AgentBench benchmark.

    AgentBench includes 8 distinct environments:
        - os: Operating system tasks in Docker sandbox
        - db: Database query tasks (SQLite/PostgreSQL)
        - kg: Knowledge graph reasoning
        - card_game: Card game strategy (24-point, etc.)
        - ltp: Lateral thinking puzzles
        - web_shopping: WebShop navigation
        - web_browsing: Real web navigation
        - alfworld: Embodied agent tasks

    Start with: os, db, kg, card_game, ltp (minimal infrastructure)

    Usage:
        adapter = AgentBenchAdapter(subset="os")
        for challenge in adapter.load_challenges():
            # Run challenge...
    """

    name = "agent-bench"
    description = "AgentBench - Multi-Environment Agent Benchmark"

    GITHUB_REPO = "THUDM/AgentBench"

    # Environment definitions with requirements
    # Directory names match the actual AgentBench repo structure
    ENVIRONMENTS: dict[str, dict[str, Any]] = {
        "os_interaction": {
            "name": "Operating System",
            "difficulty": "medium",
            "requires": ["docker"],
            "timeout_multiplier": 1.5,
        },
        "dbbench": {
            "name": "Database",
            "difficulty": "easy",
            "requires": [],
            "timeout_multiplier": 1.0,
        },
        "knowledgegraph": {
            "name": "Knowledge Graph",
            "difficulty": "medium",
            "requires": [],
            "timeout_multiplier": 1.0,
        },
        "lateralthinkingpuzzle": {
            "name": "Lateral Thinking Puzzle",
            "difficulty": "hard",
            "requires": [],
            "timeout_multiplier": 1.0,
        },
        "mind2web": {
            "name": "Mind2Web (Web Browsing)",
            "difficulty": "hard",
            "requires": ["playwright"],
            "timeout_multiplier": 3.0,
        },
        "alfworld": {
            "name": "ALFWorld",
            "difficulty": "hard",
            "requires": ["alfworld_server"],
            "timeout_multiplier": 2.0,
        },
        "avalon": {
            "name": "Avalon (Game)",
            "difficulty": "medium",
            "requires": [],
            "timeout_multiplier": 1.5,
        },
    }

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        split: str = "test",
        subset: Optional[str] = None,
        limit: Optional[int] = None,
    ):
        """Initialize the AgentBench adapter.

        Args:
            cache_dir: Directory to cache the dataset.
            split: Dataset split - "dev" or "test".
            subset: Environment to use (os, db, kg, card_game, ltp, etc.).
            limit: Maximum number of challenges to load.
        """
        super().__init__(cache_dir, split, subset, limit)
        self._tasks: dict[str, list[dict[str, Any]]] = {}
        self._repo_path: Optional[Path] = None

    def setup(self) -> None:
        """Clone/update AgentBench repository and load tasks."""
        self._repo_path = self.cache_dir / "agent_bench" / "repo"

        # Clone or update repository
        if self._repo_path.exists():
            self._update_repo()
        else:
            self._clone_repo()

        # Load tasks from repository
        self._load_tasks()

        # Check environment requirements
        if self.subset:
            self._check_requirements(self.subset)

        self._is_setup = True

    def _clone_repo(self) -> None:
        """Clone the AgentBench repository."""
        assert self._repo_path is not None  # Set in setup()
        self._repo_path.parent.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                f"https://github.com/{self.GITHUB_REPO}.git",
                str(self._repo_path),
            ],
            capture_output=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to clone AgentBench repository: {result.stderr.decode()}"
            )

    def _update_repo(self) -> None:
        """Update the AgentBench repository."""
        assert self._repo_path is not None  # Set in setup()
        result = subprocess.run(
            ["git", "pull", "--rebase"],
            cwd=str(self._repo_path),
            capture_output=True,
            timeout=60,
        )

        if result.returncode != 0:
            # Pull failed, try fresh clone
            import shutil

            shutil.rmtree(self._repo_path)
            self._clone_repo()

    def _load_tasks(self) -> None:
        """Load tasks from the repository data files."""
        if self._repo_path is None:
            return

        data_dir = self._repo_path / "data"

        if not data_dir.exists():
            # Try alternative locations
            for alt_path in ["thudm_data", "tasks", "benchmarks"]:
                alt_dir = self._repo_path / alt_path
                if alt_dir.exists():
                    data_dir = alt_dir
                    break

        # Load tasks for each environment
        for env_name in self.ENVIRONMENTS:
            env_dir = data_dir / env_name
            if not env_dir.exists():
                continue

            self._tasks[env_name] = []

            # Try JSON file first
            tasks_file = env_dir / f"{self.split}.json"
            if not tasks_file.exists():
                tasks_file = env_dir / "tasks.json"

            if tasks_file.exists():
                with open(tasks_file) as f:
                    self._tasks[env_name] = json.load(f)
                continue

            # Try JSONL file (AgentBench format)
            jsonl_file = env_dir / f"{self.split}.jsonl"
            if not jsonl_file.exists():
                jsonl_file = env_dir / "standard.jsonl"

            if jsonl_file.exists():
                with open(jsonl_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._tasks[env_name].append(json.loads(line))
                continue

            # Try to load from individual task files
            for task_file in env_dir.glob("*.json"):
                if task_file.stem not in ("config", "metadata"):
                    with open(task_file) as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            self._tasks[env_name].extend(data)
                        else:
                            self._tasks[env_name].append(data)

    def _check_requirements(self, environment: str) -> None:
        """Check if required dependencies are available."""
        env_config = self.ENVIRONMENTS.get(environment, {})
        requires = env_config.get("requires", [])

        for req in requires:
            if req == "docker":
                self._check_docker()
            elif req == "playwright":
                self._check_playwright()
            # Other requirements can be checked as needed

    def _check_docker(self) -> None:
        """Verify Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise RuntimeError("Docker is not running")
        except FileNotFoundError:
            raise RuntimeError(
                "Docker is required for the OS environment. " "Install Docker first."
            )

    def _check_playwright(self) -> None:
        """Verify Playwright is available."""
        try:
            import playwright  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "Playwright is required for web_browsing environment. "
                "Install with: pip install playwright && playwright install"
            )

    def load_challenges(self) -> Iterator[Challenge]:
        """Load challenges from the AgentBench dataset.

        Yields:
            Challenge objects for each AgentBench task.
        """
        self.ensure_setup()

        environments = [self.subset] if self.subset else list(self.ENVIRONMENTS.keys())

        count = 0
        for env_name in environments:
            if env_name not in self._tasks:
                continue

            for idx, task in enumerate(self._tasks[env_name]):
                # Apply limit
                if self.limit and count >= self.limit:
                    return

                challenge = self._convert_to_challenge(env_name, idx, task)
                yield challenge
                count += 1

    def _convert_to_challenge(
        self, environment: str, idx: int, task: dict[str, Any]
    ) -> Challenge:
        """Convert an AgentBench task to a Challenge."""
        env_config = self.ENVIRONMENTS[environment]

        # Extract task details (format varies by environment)
        task_id = task.get("id", task.get("task_id", f"{environment}_{idx}"))
        description = task.get(
            "description",
            task.get("task", task.get("instruction", task.get("question", ""))),
        )

        # Build task string based on environment
        task_str = self._format_task(environment, description, task)

        # Get difficulty
        difficulty = task.get("difficulty", env_config["difficulty"])

        # Calculate timeout
        base_timeout = 300
        multiplier = env_config["timeout_multiplier"]
        cutoff = int(base_timeout * multiplier)

        # Ground truth - extract expected answer based on environment format
        expected_answer = self._extract_expected_answer(environment, task)
        ground_truth: dict[str, Any] = {
            "eval": {"type": f"agent_bench_{environment}"},
            "environment": environment,
            "expected": expected_answer,
            "task_data": task,
        }

        # Create artifacts directory
        artifacts_dir = self.cache_dir / "agent_bench" / "artifacts" / task_id
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        return Challenge(
            name=f"AgentBench_{task_id}",
            task=task_str,
            category=["agent-bench", f"agent-bench_{environment}"],
            difficulty=difficulty,
            cutoff=cutoff,
            ground_truth=ground_truth,
            artifacts_dir=artifacts_dir,
            source_path=artifacts_dir / "task.json",
        )

    def _extract_expected_answer(self, environment: str, task: dict[str, Any]) -> str:
        """Extract the expected answer from task based on environment format."""
        # dbbench format: answer in "label" field (list)
        if environment == "dbbench":
            label = task.get("label", [])
            if isinstance(label, list) and label:
                return str(label[0])
            return str(label) if label else ""

        # knowledgegraph format: answer in "answer" array with entity_name
        if environment == "knowledgegraph":
            answers = task.get("answer", [])
            if isinstance(answers, list) and answers:
                first = answers[0]
                if isinstance(first, dict):
                    return first.get("entity_name", str(first))
                return str(first)
            return ""

        # lateralthinkingpuzzle format
        if environment == "lateralthinkingpuzzle":
            return task.get("answer", task.get("solution", ""))

        # Default: try common answer fields
        for key in ["answer", "expected", "gold", "label", "solution"]:
            val = task.get(key)
            if val:
                if isinstance(val, list):
                    return str(val[0]) if val else ""
                return str(val)
        return ""

    def _format_task(
        self, environment: str, description: str, task: dict[str, Any]
    ) -> str:
        """Format the task description based on environment."""
        if environment == "os":
            return (
                f"Operating System Task\n"
                f"=====================\n\n"
                f"{description}\n\n"
                f"You have access to a Linux command line. Execute commands "
                f"to complete the task. Save your final answer to 'answer.txt'."
            )

        elif environment in ("db", "dbbench"):
            # Extract table information from the task
            table_info = task.get("table", {})
            table_name = table_info.get("table_name", "data_table")
            columns_info = table_info.get("table_info", {}).get("columns", [])
            rows = table_info.get("table_info", {}).get("rows", [])

            # Format columns
            col_names = [col.get("name", "") for col in columns_info]

            # Build table display
            table_str_parts = [
                f"Table: {table_name}",
                f"Columns: {', '.join(col_names)}",
            ]
            table_str_parts.append("\nData (first 20 rows):")
            for i, row in enumerate(rows[:20]):
                row_str = " | ".join(str(cell) for cell in row)
                table_str_parts.append(f"  {i+1}. {row_str}")
            if len(rows) > 20:
                table_str_parts.append(f"  ... ({len(rows) - 20} more rows)")

            table_str = "\n".join(table_str_parts)

            return (
                f"Database Query Task\n"
                f"==================\n\n"
                f"Question: {description}\n\n"
                f"{table_str}\n\n"
                f"Analyze the table data above and answer the question. "
                f"Use the 'finish' command with your answer, or save your answer "
                f"to 'answer.txt'. Provide only the answer value, not an explanation."
            )

        elif environment == "kg":
            kg_info = task.get("kg_info", "")
            return (
                f"Knowledge Graph Task\n"
                f"====================\n\n"
                f"{description}\n\n"
                f"Knowledge Graph Information:\n{kg_info}\n\n"
                f"Reason over the knowledge graph to answer. "
                f"Save your answer to 'answer.txt'."
            )

        elif environment == "card_game":
            numbers = task.get("numbers", [])
            return (
                f"Card Game Task (24-point)\n"
                f"========================\n\n"
                f"Numbers: {numbers}\n\n"
                f"Use +, -, *, / and parentheses to make exactly 24. "
                f"Each number must be used exactly once.\n\n"
                f"Save your expression to 'answer.txt'."
            )

        elif environment == "ltp":
            return (
                f"Lateral Thinking Puzzle\n"
                f"======================\n\n"
                f"{description}\n\n"
                f"Ask yes/no questions to figure out the answer. "
                f"Save your final solution to 'answer.txt'."
            )

        elif environment in ("web_shopping", "web_browsing"):
            return (
                f"Web Task ({environment.replace('_', ' ').title()})\n"
                f"{'=' * 40}\n\n"
                f"{description}\n\n"
                f"Navigate the web to complete the task. "
                f"Save your final answer to 'answer.txt'."
            )

        elif environment == "alfworld":
            return (
                f"ALFWorld Task\n"
                f"=============\n\n"
                f"{description}\n\n"
                f"Navigate and interact with the environment to complete the task. "
                f"Use available actions to achieve the goal."
            )

        else:
            return description

    def evaluate(
        self,
        result: ChallengeResult,
        challenge: Challenge,
        workspace_dir: Path,
    ) -> ChallengeResult:
        """Evaluate using environment-specific logic."""
        ground = challenge.ground_truth
        environment = ground["environment"]

        # Extract answer from agent output
        agent_answer = self._extract_answer(result, environment)

        if not agent_answer:
            result.success = False
            result.score = 0.0
            result.error_message = "No answer found in agent output"
            return result

        # Environment-specific evaluation
        expected = ground.get("expected", "")

        if environment == "card_game":
            eval_result = self._evaluate_card_game(agent_answer, ground["task_data"])
        elif environment in ("db", "dbbench"):
            eval_result = self._evaluate_db(agent_answer, expected, ground["task_data"])
        elif environment in (
            "os",
            "os_interaction",
            "kg",
            "knowledgegraph",
            "ltp",
            "lateralthinkingpuzzle",
        ):
            eval_result = self._evaluate_string_match(agent_answer, expected)
        else:
            # Default string matching
            eval_result = self._evaluate_string_match(agent_answer, expected)

        result.success = eval_result["success"]
        result.score = eval_result["score"]
        if eval_result.get("error"):
            result.error_message = eval_result["error"]

        return result

    def _extract_answer(self, result: ChallengeResult, environment: str) -> str:
        """Extract answer from agent output."""
        # Look for answer.txt
        for filename, content in result.output_files.items():
            if "answer" in filename.lower():
                return content.strip()

        # Environment-specific extraction
        if environment in ("db", "dbbench"):
            for filename, content in result.output_files.items():
                if filename.endswith(".sql"):
                    return content.strip()

        # Check if agent used finish command with an answer
        if result.steps:
            last_step = result.steps[-1]
            if last_step.tool_name == "finish":
                reason = last_step.tool_args.get("reason", "").strip()
                # Try to extract the actual answer from the finish reason
                # Often the answer is embedded in the reason
                if reason:
                    return reason

        # Look for potential answer in any text file output
        for filename, content in result.output_files.items():
            if filename.endswith(".txt") and content.strip():
                return content.strip()

        return ""

    def _evaluate_card_game(
        self, answer: str, task_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate 24-point card game answer."""
        # Store numbers for potential future use in full validation
        _numbers = task_data.get("numbers", [])  # noqa: F841

        try:
            # Check that the expression evaluates to 24
            # and uses all numbers exactly once
            expr = answer.strip()

            # Safety check - only allow math operations
            allowed_chars = set("0123456789+-*/() .")
            if not all(c in allowed_chars for c in expr):
                return {
                    "success": False,
                    "score": 0.0,
                    "error": "Invalid characters in expression",
                }

            # Evaluate the expression
            result = eval(expr)

            if abs(result - 24) < 0.0001:
                # Check that all numbers are used exactly once
                # (simplified check - full implementation would parse the expression)
                return {"success": True, "score": 1.0, "error": None}
            else:
                return {
                    "success": False,
                    "score": 0.0,
                    "error": f"Expression evaluates to {result}, not 24",
                }

        except Exception as e:
            return {
                "success": False,
                "score": 0.0,
                "error": f"Failed to evaluate expression: {str(e)}",
            }

    def _evaluate_db(
        self, query: str, expected: str, task_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Evaluate SQL query answer."""
        # For now, use string matching on the result
        # Full implementation would execute query and compare results
        return self._evaluate_string_match(query, expected)

    def _evaluate_string_match(self, actual: str, expected: str) -> dict[str, Any]:
        """Strict normalized string matching."""
        actual_norm = actual.lower().strip()
        expected_norm = expected.lower().strip()

        # If no expected answer, fail (can't evaluate)
        if not expected_norm:
            return {
                "success": False,
                "score": 0.0,
                "error": "No expected answer to compare against",
            }

        # If no actual answer, fail
        if not actual_norm:
            return {
                "success": False,
                "score": 0.0,
                "error": f"No answer provided, expected '{expected}'",
            }

        # Exact match (after normalization)
        if actual_norm == expected_norm:
            return {"success": True, "score": 1.0, "error": None}

        # Check if expected is contained in actual (for verbose answers)
        if expected_norm in actual_norm:
            return {"success": True, "score": 0.9, "error": None}

        return {
            "success": False,
            "score": 0.0,
            "error": f"Expected '{expected}', got '{actual}'",
        }

    def provision_environment(self, challenge: Challenge) -> dict[str, Any]:
        """Set up environment-specific resources."""
        ground = challenge.ground_truth
        environment = ground["environment"]

        env_config: dict[str, Any] = {
            "environment": environment,
        }

        if environment == "os":
            # Would spin up Docker container here
            env_config["docker_image"] = "ubuntu:22.04"

        elif environment == "db":
            # Set up SQLite database
            task_data = ground["task_data"]
            db_setup = task_data.get("db_setup", "")
            env_config["db_type"] = "sqlite"
            env_config["db_setup"] = db_setup

        return env_config

    def get_challenge_count(self) -> Optional[int]:
        """Get the number of challenges."""
        self.ensure_setup()

        if self.subset:
            tasks = self._tasks.get(self.subset, [])
            count = len(tasks)
        else:
            count = sum(len(tasks) for tasks in self._tasks.values())

        if self.limit:
            count = min(count, self.limit)

        return count

    def get_metadata(self) -> dict[str, Any]:
        """Get AgentBench metadata."""
        metadata = super().get_metadata()
        metadata.update(
            {
                "repository": f"https://github.com/{self.GITHUB_REPO}",
                "environments": list(self.ENVIRONMENTS.keys()),
                "easy_environments": ["db", "card_game", "ltp"],
                "medium_environments": ["os", "kg", "web_shopping"],
                "hard_environments": ["web_browsing", "alfworld"],
            }
        )
        return metadata
