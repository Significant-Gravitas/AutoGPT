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
  it("shows a subtitle and no accordion while the tool is streaming", () => {
    const { container } = render(
      <GenericTool part={makePart({ state: "input-streaming" })} />,
    );
    expect(screen.queryByRole("button")).toBeNull();
    expect(container.textContent).toContain("Running");
  });

  it("shows the subtitle plus an accordion whose description is the exit status", () => {
    render(
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
    // Description shows the exit status (not the command — the subtitle row
    // above already shows what ran).
    expect(trigger.textContent).toContain("exit 1");
    expect(trigger.textContent).not.toContain(
      'echo "starting simulation run 2"',
    );
  });

  it("labels a timed-out command 'timed out' in the accordion description", () => {
    render(
      <GenericTool
        part={makePart({
          state: "output-available",
          input: { command: "sleep 120" },
          output: {
            exit_code: -1,
            timed_out: true,
            stderr: "Timed out after 120s",
          },
        })}
      />,
    );

    const trigger = screen.getByRole("button", { expanded: false });
    expect(trigger.textContent).toContain("Command timed out");
    expect(trigger.textContent).toContain("timed out");
    expect(trigger.textContent).not.toContain("sleep 120");
  });
});
