# Website and Documentation Site 📰📖
Check out *https://agpt.co*, the official news & updates site for Auto-GPT!
The documentation also has a place here, at *https://docs.agpt.co*

# For contributors 👷🏼
Since releasing v0.3.0, we are working on re-architecting the Auto-GPT core to make
it more extensible and to make room for structural performance-oriented R&D.
In the meantime, we have less time to process incoming pull requests and issues,
so we focus on high-value contributions:
 * significant bugfixes
 * *major* improvements to existing functionality and/or docs (so no single-typo fixes)
 * contributions that help us with re-architecture and other roadmapped items
We have to be somewhat selective in order to keep making progress, but this does not
mean you can't contribute. Check out the contribution guide on our wiki:
https://github.com/Significant-Gravitas/Auto-GPT/wiki/Contributing

# 🚀 v0.3.1 Release 🚀
Over a week and 47 pull requests have passed since v0.3.0, and we are happy to announce
the release of v0.3.1!

Highlights and notable changes since v0.2.2:

## Changes to Docker configuration 🐋
 * The workdir has been changed from */home/appuser* to */app*.
    Be sure to update any volume mounts accordingly!
 * Docker-compose 1.29.0 is now required.

## Logging 🧾
 * Log functionality has been improved for better understanding
    and easier summarization.
 * All LLM interactions are now logged to logs/DEBUG, to help with
    debugging and development.

## Other
 * Edge browser is now supported by the `browse_website` command.
 * Sets of commands can now be disabled using DISABLED_COMMAND_CATEGORIES in .env.

# ⚠️ Command `send_tweet` is REMOVED
Twitter functionality (and more) is now covered by plugins, see [Plugin support 🔌]

## Plugin support 🔌
Auto-GPT now has support for plugins! With plugins, you can extend Auto-GPT's abilities,
adding support for third-party services and more.
See https://github.com/Significant-Gravitas/Auto-GPT-Plugins for instructions and
available plugins. Specific plugins can be allowlisted/denylisted in .env.

## Memory backend deprecation ⚠️
The Milvus, Pinecone and Weaviate memory backends were rendered incompatible
by work on the memory system, and have been removed in `master`. The Redis
memory store was also temporarily removed but we aim to merge a new implementation
before the next release.
Whether built-in support for the others will be added back in the future is subject to
discussion, feel free to pitch in: https://github.com/Significant-Gravitas/Auto-GPT/discussions/4280
