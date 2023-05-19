# Website and Documentation Site 📰📖
Check out *https://agpt.co*, the official news & updates site for Auto-GPT!
The documentation also has a place here, at *https://docs.agpt.co*

# 🚀 v0.3.1 Release 🚀
Over a week and 47 pull requests have passed since v0.3.0, and we are happy to announce
the release of v0.3.1!

Highlights and notable changes in this release:

## Changes to Docker configuration 🐋
The workdir has been changed from */home/appuser* to */app*.
Be sure to update any volume mounts accordingly!

# ⚠️ Command `send_tweet` is DEPRECATED, and will be removed in v0.4.0 ⚠️
Twitter functionality (and more) is now covered by plugins, see [Plugin support 🔌]

## Documentation
- Docker-compose 1.29.0 is now required, as documented.
- Path to the workspace directory in the setup guide has been corrected.
- Memory setup links have been updated.

## Logs
- Log functionality has been improved for better understanding and easier summarization.
- User input is now logged in the logs/Debug Folder.

## Other 
- Edge browser support has been added using EdgeChromiumDriverManager.
- Users now have the ability to disable commands via the .env file.
- Run scripts for both Windows (.bat) and Unix (.sh) have been updated.

## BugFix
- DuckDuckGo dependency has been updated, with a minimum version set to 2.9.5.
- Package versions parsing has been enabled for forced upgrades.
- Docker volume mounts have been fixed.
- A fix was made to the plugin.post_planning call.
- A selenium driver object reference bug in the browsing results was fixed.
- JSON error in summary_memory.py has been handled.
- Dockerfile has been updated to add missing scripts and plugins directories.

## CI
- The CI pipeline has been tightened up for improved performance.
- pytest-xdist Plugin has been integrated for parallel and concurrent testing.
- Tests have been conducted for a new CI pipeline.
- A code owners policy has been added.
- Test against Python 3.10 (not 3.10 + 3.11) to halve the number of tests that are executed.

## Plugin support 🔌
Auto-GPT now has support for plugins! With plugins, you can extend Auto-GPT's abilities,
adding support for third-party services and more.
See https://github.com/Significant-Gravitas/Auto-GPT-Plugins for instructions and available plugins.
Denylist handling for plugins is now available.

*From now on, we will be focusing on major improvements* rather
than bugfixes, as we feel stability has reached a reasonable level. Most remaining
issues relate to limitations in prompt generation and the memory system, which will be
the focus of our efforts for the next release.

