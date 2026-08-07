import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import {
  getAnimationText,
  getSetupTriggerToolOutput,
  isSetupTriggerErrorOutput,
  isSetupTriggerNeedLoginOutput,
  isSetupTriggerSetupRequirementsOutput,
  isTriggerConfigRequiredOutput,
  isTriggerSetupOutput,
  ToolIcon,
} from "../helpers";

const triggerSetup = {
  type: "trigger_setup",
  message: "Trigger ready.",
  preset_id: "p1",
  library_agent_id: "lib-1",
  library_agent_link: "/library/agents/lib-1",
  name: "My Trigger",
  is_active: true,
  provider: "generic_webhook",
  manual_setup_required: false,
} as const;

const setupRequirements = {
  type: "setup_requirements",
  message: "Choose an account.",
  setup_info: { agent_name: "My Agent" },
} as const;

const configRequired = {
  type: "trigger_config_required",
  message: "Which repo?",
  missing_config: ["repo"],
  config_schema: {},
} as const;

const needLogin = { type: "need_login", message: "Sign in." } as const;
const errorOut = {
  type: "error",
  message: "Something failed",
  error: "boom",
} as const;

describe("type guards", () => {
  it("isSetupTriggerSetupRequirementsOutput matches by type and by setup_info", () => {
    expect(isSetupTriggerSetupRequirementsOutput(setupRequirements)).toBe(true);
    expect(
      isSetupTriggerSetupRequirementsOutput({
        setup_info: { agent_name: "x" },
      } as never),
    ).toBe(true);
    expect(isSetupTriggerSetupRequirementsOutput(triggerSetup as never)).toBe(
      false,
    );
  });

  it("isTriggerSetupOutput matches by type and by manual_setup_required", () => {
    expect(isTriggerSetupOutput(triggerSetup)).toBe(true);
    expect(isTriggerSetupOutput({ manual_setup_required: true } as never)).toBe(
      true,
    );
    expect(isTriggerSetupOutput(setupRequirements as never)).toBe(false);
  });

  it("isTriggerConfigRequiredOutput matches by type and by missing_config", () => {
    expect(isTriggerConfigRequiredOutput(configRequired)).toBe(true);
    expect(isTriggerConfigRequiredOutput({ missing_config: [] } as never)).toBe(
      true,
    );
    expect(isTriggerConfigRequiredOutput(triggerSetup as never)).toBe(false);
  });

  it("isSetupTriggerNeedLoginOutput matches only need_login type", () => {
    expect(isSetupTriggerNeedLoginOutput(needLogin)).toBe(true);
    expect(isSetupTriggerNeedLoginOutput(errorOut as never)).toBe(false);
  });

  it("isSetupTriggerErrorOutput matches by type and by error key", () => {
    expect(isSetupTriggerErrorOutput(errorOut)).toBe(true);
    expect(isSetupTriggerErrorOutput({ error: "x" } as never)).toBe(true);
    expect(isSetupTriggerErrorOutput(needLogin as never)).toBe(false);
  });
});

describe("getSetupTriggerToolOutput / parseOutput", () => {
  it("returns null for a non-object part", () => {
    expect(getSetupTriggerToolOutput(null)).toBeNull();
    expect(getSetupTriggerToolOutput("nope")).toBeNull();
  });

  it("returns null when there is no output", () => {
    expect(getSetupTriggerToolOutput({})).toBeNull();
    expect(getSetupTriggerToolOutput({ output: null })).toBeNull();
    expect(getSetupTriggerToolOutput({ output: "" })).toBeNull();
    expect(getSetupTriggerToolOutput({ output: "   " })).toBeNull();
  });

  it("parses a JSON string output", () => {
    const parsed = getSetupTriggerToolOutput({
      output: JSON.stringify(triggerSetup),
    });
    expect(parsed && isTriggerSetupOutput(parsed)).toBe(true);
  });

  it("returns null for an invalid JSON string", () => {
    expect(getSetupTriggerToolOutput({ output: "{not json" })).toBeNull();
  });

  it("recognises each object shape", () => {
    expect(
      getSetupTriggerToolOutput({ output: setupRequirements }),
    ).not.toBeNull();
    expect(getSetupTriggerToolOutput({ output: triggerSetup })).not.toBeNull();
    expect(
      getSetupTriggerToolOutput({ output: configRequired }),
    ).not.toBeNull();
    expect(
      getSetupTriggerToolOutput({ output: { details: "x" } }),
    ).not.toBeNull();
    expect(getSetupTriggerToolOutput({ output: needLogin })).not.toBeNull();
  });

  it("returns null for an unrecognised object type", () => {
    expect(getSetupTriggerToolOutput({ output: { foo: "bar" } })).toBeNull();
  });
});

describe("getAnimationText", () => {
  it("returns the setup phrase while streaming", () => {
    expect(getAnimationText({ state: "input-streaming" })).toBe(
      "Setting up the trigger",
    );
    expect(getAnimationText({ state: "input-available" })).toBe(
      "Setting up the trigger",
    );
  });

  it("falls back to the setup phrase when output is unparseable", () => {
    expect(
      getAnimationText({ state: "output-available", output: "{bad" }),
    ).toBe("Setting up the trigger");
  });

  it("describes a trigger_setup output by name", () => {
    expect(
      getAnimationText({ state: "output-available", output: triggerSetup }),
    ).toBe('Trigger "My Trigger" set up');
  });

  it("describes a setup_requirements output by agent name", () => {
    expect(
      getAnimationText({
        state: "output-available",
        output: setupRequirements,
      }),
    ).toBe('Setup needed for "My Agent"');
  });

  it("describes a trigger_config_required output", () => {
    expect(
      getAnimationText({ state: "output-available", output: configRequired }),
    ).toBe("Trigger configuration needed");
  });

  it("describes a need_login output", () => {
    expect(
      getAnimationText({ state: "output-available", output: needLogin }),
    ).toBe("Sign in required to set up the trigger");
  });

  it("returns the error phrase for an error output", () => {
    expect(
      getAnimationText({ state: "output-available", output: errorOut }),
    ).toBe("Something went wrong");
  });

  it("returns the error phrase for output-error state", () => {
    expect(getAnimationText({ state: "output-error" })).toBe(
      "Something went wrong",
    );
  });

  it("returns the setup phrase for an unknown state", () => {
    expect(getAnimationText({ state: "unknown" as never })).toBe(
      "Setting up the trigger",
    );
  });
});

describe("ToolIcon", () => {
  it("renders the error icon when isError", () => {
    const { container } = render(<ToolIcon isError />);
    expect(container.querySelector("svg")).not.toBeNull();
  });

  it("renders the loader while streaming", () => {
    const { container } = render(<ToolIcon isStreaming />);
    expect(container.firstChild).not.toBeNull();
  });

  it("renders the default webhook icon otherwise", () => {
    const { container } = render(<ToolIcon />);
    expect(container.querySelector("svg")).not.toBeNull();
  });
});
