import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  AccordionIcon,
  formatMaybeJson,
  getAccordionMeta,
  getAnimationText,
  getRunAgentToolOutput,
  isRunAgentAgentDetailsOutput,
  isRunAgentAgentOutputResponse,
  isRunAgentErrorOutput,
  isRunAgentExecutionStartedOutput,
  isRunAgentNeedLoginOutput,
  isRunAgentSetupRequirementsOutput,
  type RunAgentToolOutput,
  ToolIcon,
} from "../helpers";

const executionStarted = {
  type: "execution_started",
  execution_id: "exec-1",
  graph_name: "My Graph",
  status: "running",
} as unknown as RunAgentToolOutput;

const agentDetails = {
  type: "agent_details",
  message: "Inputs needed",
  agent: { id: "g1", name: "Summariser", inputs: {} },
} as unknown as RunAgentToolOutput;

const agentDetailsWebhook = {
  type: "agent_details",
  message: "Webhook trigger",
  agent: {
    id: "g2",
    name: "PR Notifier",
    inputs: {},
    trigger_info: { provider: "github" },
  },
} as unknown as RunAgentToolOutput;

const setupRequirements = {
  type: "setup_requirements",
  message: "Setup needed",
  setup_info: {
    agent_name: "My Agent",
    user_readiness: { missing_credentials: { gh: {}, openai: {} } },
  },
} as unknown as RunAgentToolOutput;

const setupRequirementsNoCreds = {
  type: "setup_requirements",
  message: "All set, ready to run",
  setup_info: { agent_name: "Ready Agent", user_readiness: {} },
} as unknown as RunAgentToolOutput;

const agentOutput = {
  type: "agent_output",
  agent_name: "Output Agent",
} as unknown as RunAgentToolOutput;

const needLogin = {
  type: "need_login",
  message: "Sign in",
} as unknown as RunAgentToolOutput;

const errorOut = {
  type: "error",
  error: "boom",
} as unknown as RunAgentToolOutput;

describe("type guards", () => {
  it("identify each output shape", () => {
    expect(isRunAgentExecutionStartedOutput(executionStarted)).toBe(true);
    expect(isRunAgentAgentDetailsOutput(agentDetails)).toBe(true);
    expect(isRunAgentAgentOutputResponse(agentOutput)).toBe(true);
    expect(isRunAgentSetupRequirementsOutput(setupRequirements)).toBe(true);
    expect(isRunAgentNeedLoginOutput(needLogin)).toBe(true);
    expect(isRunAgentErrorOutput(errorOut)).toBe(true);
  });

  it("match by duck-typed keys as well as the type field", () => {
    expect(
      isRunAgentExecutionStartedOutput({ execution_id: "x" } as never),
    ).toBe(true);
    expect(isRunAgentAgentDetailsOutput({ agent: {} } as never)).toBe(true);
    expect(isRunAgentSetupRequirementsOutput({ setup_info: {} } as never)).toBe(
      true,
    );
    expect(isRunAgentErrorOutput({ error: "x" } as never)).toBe(true);
  });
});

describe("getRunAgentToolOutput / parseOutput", () => {
  it("returns null for non-objects and empty output", () => {
    expect(getRunAgentToolOutput(null)).toBeNull();
    expect(getRunAgentToolOutput("nope")).toBeNull();
    expect(getRunAgentToolOutput({})).toBeNull();
    expect(getRunAgentToolOutput({ output: "" })).toBeNull();
    expect(getRunAgentToolOutput({ output: "  " })).toBeNull();
  });

  it("parses JSON-string outputs", () => {
    const parsed = getRunAgentToolOutput({
      output: JSON.stringify(executionStarted),
    });
    expect(parsed && isRunAgentExecutionStartedOutput(parsed)).toBe(true);
  });

  it("returns null for invalid JSON strings", () => {
    expect(getRunAgentToolOutput({ output: "{bad" })).toBeNull();
  });

  it("recognises objects by known type and by duck-typed keys", () => {
    expect(getRunAgentToolOutput({ output: agentDetails })).not.toBeNull();
    expect(getRunAgentToolOutput({ output: { details: "x" } })).not.toBeNull();
    expect(getRunAgentToolOutput({ output: { foo: "bar" } })).toBeNull();
  });
});

describe("getAnimationText", () => {
  it("uses the agent slug while streaming", () => {
    expect(
      getAnimationText({
        state: "input-streaming",
        input: { username_agent_slug: "me/agent" },
      }),
    ).toBe('Running the agent "me/agent"');
  });

  it("uses the library agent id when no slug is present", () => {
    expect(
      getAnimationText({
        state: "input-available",
        input: { library_agent_id: "lib-9" },
      }),
    ).toBe('Running the agent "Library agent lib-9"');
  });

  it("uses the scheduling phrase when a schedule is requested", () => {
    expect(
      getAnimationText({
        state: "input-streaming",
        input: { schedule_name: "daily", cron: "0 0 * * *" },
      }),
    ).toBe("Scheduling the agent to run");
  });

  it("describes an execution_started output", () => {
    expect(
      getAnimationText({ state: "output-available", output: executionStarted }),
    ).toBe('Started "My Graph"');
  });

  it("describes a webhook-trigger agent_details output", () => {
    expect(
      getAnimationText({
        state: "output-available",
        output: agentDetailsWebhook,
      }),
    ).toBe('Webhook trigger setup for "PR Notifier"');
  });

  it("describes a non-webhook agent_details output", () => {
    expect(
      getAnimationText({ state: "output-available", output: agentDetails }),
    ).toBe('Agent inputs needed for "Summariser"');
  });

  it("describes a setup_requirements output", () => {
    expect(
      getAnimationText({
        state: "output-available",
        output: setupRequirements,
      }),
    ).toBe('Setup needed for "My Agent"');
  });

  it("describes a need_login output", () => {
    expect(
      getAnimationText({ state: "output-available", output: needLogin }),
    ).toBe("Sign in required to run agent");
  });

  it("falls back to the action phrase when output is unparseable", () => {
    expect(
      getAnimationText({ state: "output-available", output: "{bad" }),
    ).toBe("Running the agent");
  });

  it("returns the error phrase for an error output and output-error state", () => {
    expect(
      getAnimationText({ state: "output-available", output: errorOut }),
    ).toBe("Something went wrong");
    expect(getAnimationText({ state: "output-error" })).toBe(
      "Something went wrong",
    );
  });

  it("returns the action phrase for an unknown state", () => {
    expect(getAnimationText({ state: "unknown" as never })).toBe(
      "Running the agent",
    );
  });
});

describe("getAccordionMeta", () => {
  it("uses the graph name and status for execution_started", () => {
    const meta = getAccordionMeta(executionStarted);
    expect(meta.title).toBe("My Graph");
    expect(meta.description).toBe("Status: running");
  });

  it("defaults the status to started when blank", () => {
    const meta = getAccordionMeta({
      type: "execution_started",
      execution_id: "x",
      graph_name: "G",
      status: "   ",
    } as unknown as RunAgentToolOutput);
    expect(meta.description).toBe("Status: started");
  });

  it("describes the webhook-trigger agent_details branch", () => {
    const meta = getAccordionMeta(agentDetailsWebhook);
    expect(meta.title).toBe("PR Notifier");
    expect(meta.description).toBe("Webhook trigger setup");
  });

  it("describes the non-webhook agent_details branch", () => {
    const meta = getAccordionMeta(agentDetails);
    expect(meta.description).toBe("Inputs required");
  });

  it("pluralises the missing-credentials count for setup_requirements", () => {
    const meta = getAccordionMeta(setupRequirements);
    expect(meta.title).toBe("My Agent");
    expect(meta.description).toBe("Missing 2 credentials");
  });

  it("falls back to the message when no credentials are missing", () => {
    const meta = getAccordionMeta(setupRequirementsNoCreds);
    expect(meta.description).toBe("All set, ready to run");
  });

  it("describes agent_output", () => {
    const meta = getAccordionMeta(agentOutput);
    expect(meta.title).toBe("Output Agent");
    expect(meta.description).toBe("Execution completed");
  });

  it("describes need_login", () => {
    expect(getAccordionMeta(needLogin).title).toBe("Sign in required");
  });

  it("describes an error fallback", () => {
    const meta = getAccordionMeta(errorOut);
    expect(meta.title).toBe("Error");
    expect(meta.titleClassName).toContain("text-red-500");
  });
});

describe("formatMaybeJson", () => {
  it("returns strings as-is", () => {
    expect(formatMaybeJson("hello")).toBe("hello");
  });

  it("stringifies objects", () => {
    expect(formatMaybeJson({ a: 1 })).toBe('{\n  "a": 1\n}');
  });

  it("falls back to String() for non-serialisable values", () => {
    const circular: Record<string, unknown> = {};
    circular.self = circular;
    expect(formatMaybeJson(circular)).toBe("[object Object]");
  });
});

describe("icons", () => {
  it("ToolIcon renders error, streaming and default variants", () => {
    expect(render(<ToolIcon isError />).container.firstChild).not.toBeNull();
    expect(
      render(<ToolIcon isStreaming />).container.firstChild,
    ).not.toBeNull();
    expect(render(<ToolIcon />).container.firstChild).not.toBeNull();
  });

  it("AccordionIcon renders", () => {
    expect(render(<AccordionIcon />).container.firstChild).not.toBeNull();
  });
});
