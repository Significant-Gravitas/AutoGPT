# Re-architecture Notes

## Key Documents

- [Planned Agent Workflow](https://whimsical.com/agent-workflow-v2-NmnTQ8R7sVo7M3S43XgXmZ)
- [Original Architecture Diagram](https://www.figma.com/file/fwdj44tPR7ArYtnGGUKknw/Modular-Architecture?type=whiteboard&node-id=0-1) - This is sadly well out of date at this point.
- [Kanban](https://github.com/orgs/Significant-Gravitas/projects/1/views/1?filterQuery=label%3Are-arch)

## The Motivation

The `master` branch of AutoGPT is an organically grown amalgamation of many thoughts 
and ideas about agent-driven autonomous systems.  It lacks clear abstraction boundaries, 
has issues of global state and poorly encapsulated state, and is generally just hard to 
make effective changes to.  Mainly it's just a system that's hard to make changes to.  
And research in the field is moving fast, so we want to be able to try new ideas 
quickly.  

## Initial Planning

A large group of maintainers and contributors met do discuss the architectural 
challenges associated with the existing codebase. Many much-desired features (building 
new user interfaces, enabling project-specific agents, enabling multi-agent systems) 
are bottlenecked by the global state in the system. We discussed the tradeoffs between 
an incremental system transition and a big breaking version change and decided to go 
for the breaking version change. We justified this by saying:

- We can maintain, in essence, the same user experience as now even with a radical 
  restructuring of the codebase
- Our developer audience is struggling to use the existing codebase to build 
  applications and libraries of their own, so this breaking change will largely be 
  welcome.

## Primary Goals

- Separate the AutoGPT application code from the library code.
- Remove global state from the system
- Allow for multiple agents per user (with facilities for running simultaneously)
- Create a serializable representation of an Agent
- Encapsulate the core systems in abstractions with clear boundaries.

## Secondary goals

- Use existing tools to ditch any unnecessary cruft in the codebase (document loading, 
  json parsing, anything easier to replace than to port).
- Bring in the [core agent loop updates](https://whimsical.com/agent-workflow-v2-NmnTQ8R7sVo7M3S43XgXmZ)
  being developed simultaneously by @Pwuts 

# The Agent Subsystems

## Configuration

We want a lot of things from a configuration system. We lean heavily on it in the 
`master` branch to allow several parts of the system to communicate with each other.  
[Recent work](https://github.com/Significant-Gravitas/AutoGPT/pull/4737) has made it 
so that the config is no longer a singleton object that is materialized from the import 
state, but it's still treated as a 
[god object](https://en.wikipedia.org/wiki/God_object) containing all information about
the system and _critically_ allowing any system to reference configuration information 
about other parts of the system.  

### What we want

- It should still be reasonable to collate the entire system configuration in a 
  sensible way.
- The configuration should be validatable and validated.
- The system configuration should be a _serializable_ representation of an `Agent`.
- The configuration system should provide a clear (albeit very low-level) contract 
  about user-configurable aspects of the system.
- The configuration should reasonably manage default values and user-provided overrides.
- The configuration system needs to handle credentials in a reasonable way.
- The configuration should be the representation of some amount of system state, like 
  api budgets and resource usage.  These aspects are recorded in the configuration and 
  updated by the system itself.
- Agent systems should have encapsulated views of the configuration.  E.g. the memory 
  system should know about memory configuration but nothing about command configuration.

## Workspace

There are two ways to think about the workspace:

- The workspace is a scratch space for an agent where it can store files, write code, 
  and do pretty much whatever else it likes.
- The workspace is, at any given point in time, the single source of truth for what an 
  agent is.  It contains the serializable state (the configuration) as well as all 
  other working state (stored files, databases, memories, custom code).  

In the existing system there is **one** workspace.  And because the workspace holds so 
much agent state, that means a user can only work with one agent at a time.

## Memory

The memory system has been under extremely active development. 
See [#3536](https://github.com/Significant-Gravitas/AutoGPT/issues/3536) and 
[#4208](https://github.com/Significant-Gravitas/AutoGPT/pull/4208) for discussion and 
work in the `master` branch.  The TL;DR is 
that we noticed a couple of months ago that the `Agent` performed **worse** with 
permanent memory than without it.  Since then the knowledge storage and retrieval 
system has been [redesigned](https://whimsical.com/memory-system-8Ae6x6QkjDwQAUe9eVJ6w1) 
and partially implemented in the `master` branch.

## Planning/Prompt-Engineering

The planning system is the system that translates user desires/agent intentions into
language model prompts.  In the course of development, it has become pretty clear 
that `Planning` is the wrong name for this system

### What we want

- It should be incredibly obvious what's being passed to a language model, when it's
  being passed, and what the language model response is. The landscape of language 
  model research is developing very rapidly, so building complex abstractions between 
  users/contributors and the language model interactions is going to make it very 
  difficult for us to nimbly respond to new research developments.
- Prompt-engineering should ideally be exposed in a parameterizeable way to users. 
- We should, where possible, leverage OpenAI's new  
  [function calling api](https://openai.com/blog/function-calling-and-other-api-updates) 
  to get outputs in a standard machine-readable format and avoid the deep pit of 
  parsing json (and fixing unparsable json).

### Planning Strategies

The [new agent workflow](https://whimsical.com/agent-workflow-v2-NmnTQ8R7sVo7M3S43XgXmZ) 
has many, many interaction points for language models.  We really would like to not 
distribute prompt templates and raw strings all through the system. The re-arch solution 
is to encapsulate language model interactions into planning strategies. 
These strategies are defined by 

- The `LanguageModelClassification` they use (`FAST` or `SMART`)
- A function `build_prompt` that takes strategy specific arguments and constructs a 
  `LanguageModelPrompt` (a simple container for lists of messages and functions to
  pass to the language model)
- A function `parse_content` that parses the response content (a dict) into a better 
  formatted dict.  Contracts here are intentionally loose and will tighten once we have 
  at least one other language model provider.

## Resources

Resources are kinds of services we consume from external APIs.  They may have associated 
credentials and costs we need to manage.  Management of those credentials is implemented 
as manipulation of the resource configuration.  We have two categories of resources 
currently

- AI/ML model providers (including language model providers and embedding model providers, ie OpenAI)
- Memory providers (e.g. Pinecone, Weaviate, ChromaDB, etc.)

### What we want

- Resource abstractions should provide a common interface to different service providers 
  for a particular kind of service.  
- Resource abstractions should manipulate the configuration to manage their credentials 
  and budget/accounting.
- Resource abstractions should be composable over an API (e.g. I should be able to make 
  an OpenAI provider that is both a LanguageModelProvider and an EmbeddingModelProvider
  and use it wherever I need those services).

## Abilities

Along with planning and memory usage, abilities are one of the major augmentations of 
augmented language models.  They allow us to expand the scope of what language models
can do by hooking them up to code they can execute to obtain new knowledge or influence
the world.  

### What we want

- Abilities should have an extremely clear interface that users can write to.
- Abilities should have an extremely clear interface that a language model can 
  understand
- Abilities should be declarative about their dependencies so the system can inject them
- Abilities should be executable (where sensible) in an async run loop.
- Abilities should be not have side effects unless those side effects are clear in 
  their representation to an agent (e.g. the BrowseWeb ability shouldn't write a file,
  but the WriteFile ability can).

## Plugins

Users want to add lots of features that we don't want to support as first-party. 
Or solution to this is a plugin system to allow users to plug in their functionality or
to construct their agent from a public plugin marketplace.  Our primary concern in the
re-arch is to build a stateless plugin service interface and a simple implementation 
that can load plugins from installed packages or from zip files.  Future efforts will 
expand this system to allow plugins to load from a marketplace or some other kind 
of service.

### What is a Plugin

Plugins are a kind of garbage term.  They refer to a number of things.

- New commands for the agent to execute.  This is the most common usage.
- Replacements for entire subsystems like memory or language model providers
- Application plugins that do things like send emails or communicate via whatsapp
- The repositories contributors create that may themselves have multiple plugins in them.

### Usage in the existing system

The current plugin system is _hook-based_.  This means plugins don't correspond to 
kinds of objects in the system, but rather to times in the system at which we defer 
execution to them.  The main advantage of this setup is that user code can hijack 
pretty much any behavior of the agent by injecting code that supersedes the normal 
agent execution.  The disadvantages to this approach are numerous:

- We have absolutely no mechanisms to enforce any security measures because the threat 
  surface is everything.
- We cannot reason about agent behavior in a cohesive way because control flow can be
  ceded to user code at pretty much any point and arbitrarily change or break the
  agent behavior
- The interface for designing a plugin is kind of terrible and difficult to standardize
- The hook based implementation means we couple ourselves to a particular flow of 
  control (or otherwise risk breaking plugin behavior).  E.g. many of the hook targets
  in the [old workflow](https://whimsical.com/agent-workflow-VAzeKcup3SR7awpNZJKTyK) 
  are not present or mean something entirely different in the 
  [new workflow](https://whimsical.com/agent-workflow-v2-NmnTQ8R7sVo7M3S43XgXmZ).
- Etc.

### What we want

- A concrete definition of a plugin that is narrow enough in scope that we can define 
  it well and reason about how it will work in the system.
- A set of abstractions that let us define a plugin by its storage format and location 
- A service interface that knows how to parse the plugin abstractions and turn them 
  into concrete classes and objects.


## Some Notes on how and why we'll use OO in this project

First and foremost, Python itself is an object-oriented language. It's 
underlying [data model](https://docs.python.org/3/reference/datamodel.html) is built 
with object-oriented programming in mind. It offers useful tools like abstract base 
classes to communicate interfaces to developers who want to, e.g., write plugins, or 
help work on implementations. If we were working in a different language that offered 
different tools, we'd use a different paradigm.

While many things are classes in the re-arch, they are not classes in the same way. 
There are three kinds of things (roughly) that are written as classes in the re-arch:
1.  **Configuration**:  AutoGPT has *a lot* of configuration.  This configuration 
    is *data* and we use **[Pydantic](https://docs.pydantic.dev/latest/)** to manage it as 
    pydantic is basically industry standard for this stuff. It provides runtime validation 
    for all the configuration and allows us to easily serialize configuration to both basic 
    python types (dicts, lists, and primitives) as well as serialize to json, which is 
    important for us being able to put representations of agents 
    [on the wire](https://en.wikipedia.org/wiki/Wire_protocol) for web applications and 
    agent-to-agent communication. *These are essentially 
    [structs](https://en.wikipedia.org/wiki/Struct_(C_programming_language)) rather than 
    traditional classes.*
2.  **Internal Data**: Very similar to configuration, AutoGPT passes around boatloads 
    of internal data.  We are interacting with language models and language model APIs 
    which means we are handling lots of *structured* but *raw* text.  Here we also 
    leverage **pydantic** to both *parse* and *validate* the internal data and also to 
    give us concrete types which we can use static type checkers to validate against 
    and discover problems before they show up as bugs at runtime. *These are 
    essentially [structs](https://en.wikipedia.org/wiki/Struct_(C_programming_language)) 
    rather than traditional classes.*
3.  **System Interfaces**: This is our primary traditional use of classes in the 
    re-arch.  We have a bunch of systems. We want many of those systems to have 
    alternative implementations (e.g. via plugins). We use abstract base classes to 
    define interfaces to communicate with people who might want to provide those 
    plugins. We provide a single concrete implementation of most of those systems as a 
    subclass of the interface. This should not be controversial.

The approach is consistent with 
[prior](https://github.com/Significant-Gravitas/AutoGPT/issues/2458)
[work](https://github.com/Significant-Gravitas/AutoGPT/pull/2442) done by other 
maintainers in this direction.

From an organization standpoint, OO programming is by far the most popular programming 
paradigm (especially for Python). It's the one most often taught in programming classes
and the one with the most available online training for people interested in 
contributing.   

Finally, and importantly, we scoped the plan and initial design of the re-arch as a 
large group of maintainers and collaborators early on. This is consistent with the 
design we chose and no-one offered alternatives.
