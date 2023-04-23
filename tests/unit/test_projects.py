import unittest
from pathlib import Path
import tempfile
import shutil
import yaml
from autogpt.project.project_manager import ProjectManager


class TestProjectManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.project_manager = ProjectManager(project_folder=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_get_agents(self):
        # Create a test project folder with agent YAML files
        project_name = "test_project"
        agents_folder = Path(self.temp_dir) / project_name / "agents"
        agents_folder.mkdir(parents=True, exist_ok=True)
        agent_files = ["agent1.yaml", "agent2.yaml"]
        ai_names = ["AI One", "AI Two"]

        for idx, agent_file in enumerate(agent_files):
            with open(agents_folder / agent_file, "w") as file:
                yaml.dump({"ai_name": ai_names[idx]}, file)

        # Test get_agents method
        result = self.project_manager.get_agents(project_name)
        self.assertEqual(len(result), len(agent_files))

    def test_read_ai_name(self):
        # Create a test YAML file
        yaml_file = Path(self.temp_dir) / "test_agent.yaml"
        ai_name = "Test AI Name"
        with open(yaml_file, "w") as file:
            yaml.dump({"ai_name": ai_name}, file)

        # Test read_ai_name method
        result = self.project_manager.read_ai_name(yaml_file)
        self.assertEqual(result, ai_name)


if __name__ == "__main__":
    unittest.main()
