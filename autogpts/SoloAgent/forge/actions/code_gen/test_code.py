import subprocess

def cargo_test_agbenchmark_config(project_path: str):
    try:
        print (project_path)
        subprocess.run(["cargo", "test"], cwd=project_path, check=True)
    except FileNotFoundError:
        print("Cargo not found. Make sure it is installed.")
    except Exception as e:
        print(f"An error occurred: {e}")
project_path = "/Users/yassirmaknaoui/git/Soloagent/SoloAgent/autogpts/SoloAgent/agbenchmark_config/workspace/90606f1a-2b20-44e4-ad7e-0f6273d7aaab"
cargo_test_agbenchmark_config(project_path)