if not workspace_directory:
        # Check for default legacy workspace directory & prompt to migrate
        legacy_workspace_name = "auto_gpt_workspace"
        legacy_workspace_directory = None
        
        if Path(Path.cwd() / legacy_workspace_name).exists():
            legacy_workspace_directory = Path.cwd() / legacy_workspace_name
        elif Path(Path(__file__).parent / legacy_workspace_name).exists():
            legacy_workspace_directory = Path.cwd() / legacy_workspace_name 
        
        if legacy_workspace_directory:
            print("Warning: Old workspace directory found at " + str(legacy_workspace_directory))
            
            if interactive:
                if click.prompt("Would you like to migrate it to your home directory? (y/n)") == "y":
                    new_workspace_directory = Path.home() / legacy_workspace_name
                    print("Migrating workspace directory to " + str(new_workspace_directory))
                    shutil.move(legacy_workspace_directory, new_workspace_directory)
                    print("Workspace directory migrated successfully.")
                    workspace_directory = new_workspace_directory
                else:
                    print("Please move your old workspace directory to another location, for example, your home directory. The old workspace directory will be deprecated in the future.")
                    print("You can also specify a custom workspace directory with the --workspace-directory flag.")
                    workspace_directory = legacy_workspace_directory
    
    if not workspace_directory:
        if interactive:
            # TODO: Kick off interactive install, ask for workspace directory, openai key etc.
            pass  
        else:
            # TODO: Add non-interactive install, set default workspace directory to home directory
            pass