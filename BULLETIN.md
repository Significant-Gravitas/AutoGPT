# Website and Documentation Site 📰📖
Check out *https://agpt.co*, the official news & updates site for Auto-GPT!
The documentation also has a place here, at *https://docs.agpt.co*

# 🚀 v0.3.0 Release 🚀
Over a week and 275 pull requests have passed since v0.2.2, and we are happy to announce
the release of v0.3.0! *From now on, we will be focusing on major improvements* rather
than bugfixes, as we feel stability has reached a reasonable level. Most remaining
issues relate to limitations in prompt generation and the memory system, which will be
the focus of our efforts for the next release.

Highlights and notable changes in this release:

## Plugin support 🔌
Auto-GPT now has support for plugins! With plugins, you can extend Auto-GPT's abilities,
adding support for third-party services and more.
See https://github.com/Significant-Gravitas/Auto-GPT-Plugins for instructions and available plugins.

## Changes to Docker configuration 🐋
The workdir has been changed from */home/appuser* to */app*.
Be sure to update any volume mounts accordingly!

# ⚠️ Command `send_tweet` is DEPRECATED, and will be removed in v0.4.0 ⚠️
Twitter functionality (and more) is now covered by plugins, see [Plugin support 🔌]
