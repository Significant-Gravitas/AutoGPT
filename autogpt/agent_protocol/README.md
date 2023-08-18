# Autogpt Protocol Directory

# DO NOT MODIFY ANY FILES IN THIS DIRECTORY

This directory contains protocol definitions crucial for our project. The current setup is a temporary measure to allow for speedy updating of the protocol.

## Background

In an ideal scenario, we'd directly use a submodule pointing to the original repository. However, given our specific needs and to expedite our development process, we've chosen a slightly different approach.

## Process

1. **Fork and Clone**: We started by forking the original repository `e2b-dev/agent-protocol` (not `Swiftyos/agent-protocol` as previously mentioned) to have our own version. This allows us to have more control over updates and possibly any specific changes that our project might need in the future.

2. **Manual Content Integration**: Instead of adding the entire forked repository as a submodule, we've manually copied over the contents of `sdk/python/agent_protocol` into this directory. This ensures we only have the parts we need, without any additional overhead.

3. **Updates**: Any necessary updates to the protocol can be made directly in our fork, and subsequently, the required changes can be reflected in this directory.

## Credits

All credit for the original protocol definitions goes to [e2b-dev/agent-protocol](https://github.com/e2b-dev/agent-protocol). We deeply appreciate their efforts in building the protocol, and this temporary measure is in no way intended to diminish the significance of their work. It's purely a practical approach for our specific requirements at this point in our development phase.
