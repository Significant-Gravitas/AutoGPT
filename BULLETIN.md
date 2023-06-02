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

# 🚀 v0.4.0 Release 🚀
Two weeks and 76 pull requests have passed since v0.3.1, and we are happy to announce
the release of v0.4.0!

Highlights and notable changes since v0.3.1:

## ⚠️ Command `send_tweet` is REMOVED
Twitter functionality (and more) is now covered by plugins.

## Memory backend deprecation ⚠️
The Milvus, Pinecone and Weaviate memory backends were rendered incompatible
by work on the memory system, and have been removed in `master`. The Redis
memory store was also temporarily removed; we will merge a new implementation ASAP.
Whether built-in support for the others will be added back in the future is subject to
discussion, feel free to pitch in: https://github.com/Significant-Gravitas/Auto-GPT/discussions/4280

## Refactoring of global state
The config object 'singleton' has been refactored to ease the path to our re-arch.

## Further fixes and changes
Take a look at the Release Notes on Github! https://github.com/Significant-Gravitas/Auto-GPT/releases/tag/v0.4.0
