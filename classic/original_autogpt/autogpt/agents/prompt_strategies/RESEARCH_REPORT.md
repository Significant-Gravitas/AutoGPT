# Prompt Strategies Research Report

## Overview

This report documents the prompt strategies available in `classic/original_autogpt/autogpt/agents/prompt_strategies/` and provides recommendations for supporting subagents in the parallel benchmark.

## Strategy Summary

| Strategy | Token Efficiency | Best For | Subagent Potential |
|----------|------------------|----------|-------------------|
| **one_shot** | High | Simple tasks | Low - single action per turn |
| **plan_execute** | Medium | Multi-step tasks | High - plan can distribute steps |
| **reflexion** | Low | Learning from mistakes | Medium - reflections shareable |
| **rewoo** | Very High (5x) | Parallel execution | Very High - designed for parallelism |
| **tree_of_thoughts** | Low | Complex reasoning | High - branches can be parallel |

---

## 1. One-Shot Strategy (`one_shot.py`)

### Description
The simplest strategy - generates a single action proposal per turn with structured thoughts.

### Key Components
- `AssistantThoughts`: observations, reasoning, self-criticism, plan
- `OneShotAgentActionProposal`: thoughts + tool call
- Uses `FAST_MODEL` classification

### Architecture
```
[System Prompt] → [Task] → [History] → [Choose Action Instruction]
                                    ↓
                            Single Action Proposal
```

### Subagent Support: LOW
- **Rationale**: Single-agent design with no built-in parallelism
- **Potential**: Could spawn subagents for individual tool executions, but gains minimal

---

## 2. Plan-and-Execute Strategy (`plan_execute.py`)

### Description
Separates planning from execution. Creates a high-level plan, then executes steps sequentially with replanning on failure.

### Key Components
- **Phases**: `VARIABLE_EXTRACTION` → `PLANNING` → `EXECUTING` → `REPLANNING`
- `ExecutionPlan`: goal, steps, progress tracking
- `PlannedStep`: thought, tool_name, tool_arguments, status
- **PS+ Features**: Variable extraction and calculation verification (from Plan-and-Solve paper)

### Architecture
```
[PLAN Phase]     → Create numbered step list
[EXECUTE Phase]  → Execute current step → advance
[REPLAN Phase]   → On failure, regenerate plan from current state
```

### Research References
- Plan-and-Act (arxiv.org/html/2503.09572v3)
- Plan-and-Solve Prompting (arxiv.org/abs/2305.04091) - 96.3% accuracy
- Routine: Enterprise-Grade Planning (arxiv.org/html/2507.14447)

### Subagent Support: HIGH
- **Rationale**: Plan steps are independent work units
- **Recommendations**:
  1. **Parallel Step Execution**: Steps without dependencies can run in parallel
  2. **Plan Distribution**: Distribute independent plan branches to subagents
  3. **Hierarchical Planning**: Main agent plans, subagents execute steps
  4. **Replanning Coordination**: When one subagent fails, coordinator triggers replan

### Implementation Approach
```python
# Proposed subagent interface
class SubagentCoordinator:
    def distribute_plan(self, plan: ExecutionPlan) -> dict[str, PlannedStep]:
        """Assign steps to available subagents based on dependencies"""

    def await_step_completion(self, step_id: str) -> StepResult:
        """Wait for subagent to complete step"""

    def trigger_replan(self, failed_step: PlannedStep, error: str):
        """Coordinate replanning across subagents"""
```

---

## 3. Reflexion Strategy (`reflexion.py`)

### Description
Verbal reinforcement learning through explicit self-reflection. Learns from mistakes without training.

### Key Components
- **Phases**: `PROPOSING` ↔ `REFLECTING`
- `ReflexionMemory`: Episodic memory of past reflections (max 20)
- `Reflection`: action_name, result_summary, what_went_wrong, lesson_learned
- **Evaluator**: Heuristic or LLM-based result evaluation
- **Formats**: Structured JSON or verbal (free-form) reflections

### Architecture
```
[PROPOSE] → Execute Action → [EVALUATE] → [REFLECT] → Store Lesson
     ↑                                                      |
     └──────────────── Apply Lessons ───────────────────────┘
```

### Research References
- Reflexion: Verbal Reinforcement Learning (arxiv.org/abs/2303.11366) - 91% pass@1 on HumanEval
- Self-Refine: Iterative Self-Feedback (arxiv.org/abs/2303.17651)
- Self-Reflection in LLM Agents (arxiv.org/abs/2405.06682)

### Subagent Support: MEDIUM
- **Rationale**: Reflections are valuable across agents but timing is tricky
- **Recommendations**:
  1. **Shared Reflection Memory**: Central store of lessons learned
  2. **Cross-Agent Learning**: Subagent failures inform other subagents
  3. **Reflection Aggregation**: Synthesize reflections from multiple subagents
  4. **Failure Broadcasting**: When one subagent learns lesson, broadcast to others

### Implementation Approach
```python
# Proposed shared memory interface
class SharedReflexionMemory:
    def add_reflection(self, agent_id: str, reflection: Reflection):
        """Store reflection with source agent"""

    def get_relevant_reflections(self, context: str) -> list[Reflection]:
        """Retrieve relevant lessons from any agent"""

    def broadcast_lesson(self, lesson: str, severity: str):
        """Notify all agents of critical lesson"""
```

---

## 4. ReWOO Strategy (`rewoo.py`)

### Description
"Reasoning Without Observation" - generates complete plan upfront with variable placeholders, then executes all tools (potentially in parallel), then synthesizes results.

### Key Components
- **Phases**: `PLANNING` → `EXECUTING` → `SYNTHESIZING`
- `ReWOOPlan`: steps with variable names (#E1, #E2)
- `PlannedStep`: includes `depends_on` for dependency tracking
- `WorkerExecution`: tracks variable substitution and raw outputs
- **Paper Format**: `#E1 = Tool[argument]` or function format

### Architecture
```
[PLAN Phase]      → Generate full plan with #E1, #E2, etc.
                  ↓
[EXECUTE Phase]   → Run all tools (parallel if no deps)
                  ↓
[SYNTHESIZE]      → Combine all results into final response
```

### Research Reference
- ReWOO Paper (arxiv.org/abs/2305.18323) - **5x token efficiency vs ReAct**

### Subagent Support: VERY HIGH
- **Rationale**: Explicitly designed for parallel execution with dependency graph
- **Recommendations**:
  1. **Direct Parallel Execution**: `get_executable_steps()` already returns parallel-safe steps
  2. **Dependency-Aware Distribution**: Assign independent step chains to subagents
  3. **Variable Passing**: Central coordinator manages #E1, #E2 variable results
  4. **Synthesis Coordination**: Collect all subagent results, run synthesis once

### Implementation Approach
```python
# ReWOO is nearly subagent-ready
class ReWOOSubagentCoordinator:
    def __init__(self, plan: ReWOOPlan):
        self.plan = plan

    def get_parallel_batches(self) -> list[list[PlannedStep]]:
        """Group steps by dependency level for parallel execution"""
        batches = []
        remaining = set(range(len(self.plan.steps)))
        satisfied = set()

        while remaining:
            # Find all steps with satisfied dependencies
            batch = [s for i, s in enumerate(self.plan.steps)
                    if i in remaining and
                    all(d in satisfied for d in s.depends_on)]
            batches.append(batch)
            for s in batch:
                satisfied.add(s.variable_name)
                remaining.discard(self.plan.steps.index(s))
        return batches

    def assign_to_subagents(self, batch: list[PlannedStep]) -> dict[str, PlannedStep]:
        """Assign batch of independent steps to available subagents"""
```

---

## 5. Tree of Thoughts Strategy (`tree_of_thoughts.py`)

### Description
Deliberate problem solving through tree search. Generates multiple candidate thoughts, evaluates them, and searches the tree (BFS/DFS) with backtracking.

### Key Components
- **Phases**: `GENERATING` → `EVALUATING` → `SELECTING`
- `ThoughtTree`: Tree structure with parent pointers for backtracking
- `Thought`: content, score, children, categorical evaluation
- **Evaluation Modes**: Numeric (0-10) or categorical (sure/maybe/impossible)
- **Search Algorithms**: BFS or DFS
- **Branching Factor**: Configurable (default 3)

### Architecture
```
         [Root Problem]
              |
    ┌─────────┼─────────┐
    ↓         ↓         ↓
[Thought1] [Thought2] [Thought3]    ← Generate N candidates
   7.5       4.2        8.1         ← Evaluate each
              ↓                      ← Select best
         [Thought3]
              |
    ┌─────────┼─────────┐
    ↓         ↓         ↓
[Child1]  [Child2]  [Child3]        ← Recurse
```

### Research References
- Tree of Thoughts (arxiv.org/abs/2305.10601)
- ToTRL: Tree-of-Thought via Puzzles (arxiv.org/abs/2505.12717)
- Tree of Uncertain Thoughts (arxiv.org/abs/2309.07694)

### Subagent Support: HIGH
- **Rationale**: Tree branches are naturally parallel exploration paths
- **Recommendations**:
  1. **Parallel Branch Exploration**: Each subagent explores different branch
  2. **Distributed Evaluation**: Multiple subagents evaluate candidates simultaneously
  3. **Best-Path Aggregation**: Collect and compare best paths from all subagents
  4. **Parallel DFS**: Multiple subagents do depth-first exploration of different subtrees

### Implementation Approach
```python
# Parallel tree exploration
class ToTSubagentCoordinator:
    def distribute_branches(self, tree: ThoughtTree, n_agents: int) -> dict[str, Thought]:
        """Assign top N branches to N subagents"""
        candidates = tree.current_node.children
        sorted_candidates = sorted(candidates, key=lambda t: t.score, reverse=True)
        return {f"agent_{i}": c for i, c in enumerate(sorted_candidates[:n_agents])}

    def parallel_evaluate(self, candidates: list[Thought]) -> list[ThoughtEvaluation]:
        """Have subagents evaluate candidates in parallel"""

    def merge_explorations(self, results: dict[str, ThoughtTree]) -> ThoughtTree:
        """Merge subtrees explored by different subagents"""
```

---

## Recommendations for Parallel Benchmark

### Priority 1: ReWOO (Highest Value)

ReWOO should be the primary strategy for parallel benchmark because:
1. **Designed for parallelism**: Explicit dependency graph with variable placeholders
2. **5x token efficiency**: Massive cost savings at scale
3. **Minimal changes needed**: `get_executable_steps()` already identifies parallel-safe work
4. **Clear coordination model**: Central variable store (#E1, #E2) handles results

**Recommended Changes:**
```python
# Add to ReWOOPromptStrategy
def get_parallel_batches(self) -> list[list[PlannedStep]]:
    """Return steps grouped by dependency level for parallel execution"""

def assign_step_to_subagent(self, step: PlannedStep, subagent_id: str):
    """Mark step as assigned to specific subagent"""

def collect_subagent_result(self, variable_name: str, result: str, subagent_id: str):
    """Collect result from subagent and update variable store"""
```

### Priority 2: Plan-and-Execute (High Value)

Plan-and-Execute is well-suited for hierarchical coordination:
1. **Main agent plans**: Creates high-level plan
2. **Subagents execute**: Each step delegated to available subagent
3. **Coordinator replan**: On failure, main agent replans and redistributes

**Recommended Changes:**
```python
# Add to PlanExecutePromptStrategy
def mark_step_delegated(self, step_index: int, subagent_id: str):
    """Mark step as delegated to subagent"""

def receive_step_result(self, step_index: int, result: str, success: bool):
    """Process result from subagent, trigger replan if needed"""
```

### Priority 3: Tree of Thoughts (Medium Value)

ToT is valuable for exploration-heavy tasks:
1. **Parallel branch exploration**: Different subagents explore different paths
2. **Best-path selection**: Aggregate results and pick winner
3. **Backtracking coordination**: When one path fails, try another subagent's path

### Priority 4: Reflexion (Lower Priority)

Reflexion's value is in shared learning:
1. **Shared reflection store**: All subagents contribute and read lessons
2. **Cross-agent learning**: Failures in one subagent inform others

---

## Implementation Roadmap

### Phase 1: Core Subagent Infrastructure
1. Define `SubagentCoordinator` interface
2. Implement variable/result passing between agents
3. Add subagent lifecycle management (spawn, monitor, terminate)

### Phase 2: ReWOO Parallel Execution
1. Add `get_parallel_batches()` method
2. Implement parallel tool execution with variable substitution
3. Test with dependency-heavy benchmarks

### Phase 3: Plan-Execute Hierarchical Mode
1. Add step delegation interface
2. Implement result collection and replanning triggers
3. Test with multi-step tasks

### Phase 4: ToT Parallel Exploration
1. Add branch distribution logic
2. Implement tree merging from multiple subagents
3. Test with complex reasoning benchmarks

### Phase 5: Shared Reflexion Memory
1. Implement cross-agent reflection store
2. Add lesson broadcasting mechanism
3. Test learning transfer between agents

---

## Conclusion

**ReWOO is the clear winner** for parallel benchmark support due to its explicit parallelism design and 5x token efficiency. Plan-and-Execute is the second priority for hierarchical coordination patterns.

The existing codebase has strong foundations:
- `PlannedStep.depends_on` already tracks dependencies
- `ReWOOPlan.get_executable_steps()` identifies parallel-safe work
- Clear phase separation in all strategies enables coordination points

Key gaps to address:
1. No central subagent coordination layer
2. No shared state mechanism between agents
3. No result aggregation patterns

These gaps are addressable with a well-designed `SubagentCoordinator` class that wraps existing strategies.
