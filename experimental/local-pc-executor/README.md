# AutoGPT Local PC Executor

This monorepo contains the hosted platform integration for the Local PC
Executor. The installable shim, wire-protocol documentation, security model,
cross-platform implementation, and release workflow live in the companion
[autogpt-local-executor repository](https://github.com/Significant-Gravitas/autogpt-local-executor).

Keeping the executable shim in one repository prevents security fixes and wire
contracts from drifting between an embedded copy and the distributed package.
