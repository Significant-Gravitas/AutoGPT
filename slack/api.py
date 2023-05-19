import sys
import os
main_dir = os.getcwd()
sys.path.append(main_dir)

if __name__ == "__main__":
    print(sys.argv)
    print(os.getcwd())
    _, folder = sys.argv
    from autogpt.main import run_auto_gpt

    run_auto_gpt(
        continuous=True,
        continuous_limit=None,
        ai_settings=os.path.join(folder, "ai_settings.yaml"),
        prompt_settings='prompt_settings.yaml',
        skip_reprompt=False,
        speak=False,
        debug=False,
        gpt3only=False,
        gpt4only=False,
        memory_type=None,
        browser_name=None,
        allow_downloads=False,
        skip_news=True,
        workspace_directory=folder,
        install_plugin_deps=False
    )