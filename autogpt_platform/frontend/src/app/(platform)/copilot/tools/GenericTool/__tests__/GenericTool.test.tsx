import { describe, expect, it } from "vitest";
import type { ToolUIPart } from "ai";
import { render, screen } from "@/tests/integrations/test-utils";
import { GenericTool } from "../GenericTool";

function makePart(overrides: Record<string, unknown> = {}): ToolUIPart {
  return {
    type: "tool-bash_exec",
    toolCallId: "call-1",
    state: "input-streaming",
    input: { command: 'echo "hi"' },
    ...overrides,
  } as ToolUIPart;
}

describe("GenericTool", () => {
  it("shows only a status line while the tool is streaming", () => {
    const { container } = render(
      <GenericTool part={makePart({ state: "input-streaming" })} />,
    );
    expect(screen.queryByRole("button")).toBeNull();
    expect(container.textContent).toContain("Running");
  });

  it("shows only the accordion once output is available (no duplicate status line)", () => {
    const { container } = render(
      <GenericTool
        part={makePart({
          state: "output-available",
          input: { command: 'echo "starting simulation run 2"' },
          output: {
            exit_code: 1,
            stdout: "",
            stderr: "boom",
          },
        })}
      />,
    );

    const trigger = screen.getByRole("button", { expanded: false });
    expect(trigger.textContent).toContain("Command failed (exit 1)");
    expect(container.textContent).not.toContain("Command exited with code 1");
  });

  it("shows only the accordion on output-error (no duplicate status line)", () => {
    const { container } = render(
      <GenericTool
        part={makePart({
          state: "output-error",
          output: { exit_code: 2, stderr: "nope" },
        })}
      />,
    );

    const trigger = screen.getByRole("button", { expanded: false });
    expect(trigger.textContent).toContain("Command failed (exit 2)");
    expect(container.textContent).not.toContain("Command exited with code 2");
  });
});
