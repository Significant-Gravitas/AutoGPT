### Background

<!-- Clearly explain the need for these changes: -->

### Changes 🏗️

<!-- Concisely describe all of the changes made in this pull request: -->


### Testing 🔍
> [!NOTE] 
Only for the new autogpt platform, currently in autogpt_platform/

<!--
Please make sure your changes have been tested and are in good working condition. 
Here is a list of our critical paths, if you need some inspiration on what and how to test:
-->

- Create from scratch and execute an agent with at least 3 blocks
- Import an agent from file upload, and confirm it executes correctly
- Upload agent to marketplace
- Import an agent from marketplace and confirm it executes correctly
- Edit an agent from monitor, and confirm it executes correctly

### Configuration Changes 📝
> [!NOTE] 
Only for the new autogpt platform, currently in autogpt_platform/

If you're making configuration or infrastructure changes, please remember to check you've updated the related infrastructure code in the autogpt_platform/infra folder.

Examples of such changes might include: 

- Changing ports
- Adding new services that need to communicate with each other
- Secrets or environment variable changes
- New or infrastructure changes such as databases
