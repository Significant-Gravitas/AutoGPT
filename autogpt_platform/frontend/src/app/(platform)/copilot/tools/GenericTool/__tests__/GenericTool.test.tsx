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

  it("never shows the exit code in the subtitle (it lives in the accordion)", () => {
    const { container } = render(
      <GenericTool
        part={makePart({
          state: "output-available",
          input: { command: 'echo "starting simulation run 2"' },
          output: { exit_code: 1, stdout: "", stderr: "boom" },
        })}
      />,
    );
    expect(container.textContent).not.toContain("Command exited with code 1");
    expect(container.textContent).not.toContain(
      'echo "starting simulation run 2"',
    );
  });

  it("shows 'exit N · <first line of stderr>' on non-zero exit", () => {
    render(
      <GenericTool
        part={makePart({
          state: "output-available",
          input: { command: "missing-bin" },
          output: {
            exit_code: 127,
            stdout: "",
            stderr: "bash: missing-bin: command not found\n",
          },
        })}
      />,
    );
    const trigger = screen.getByRole("button", { expanded: false });
    expect(trigger.textContent).toContain("Command failed (exit 127)");
    expect(trigger.textContent).toContain(
      "exit 127 · bash: missing-bin: command not found",
    );
  });

  it("falls back to bare 'exit N' when stderr is empty", () => {
    render(
      <GenericTool
        part={makePart({
          state: "output-available",
          output: { exit_code: 2, stdout: "", stderr: "" },
        })}
      />,
    );
    const trigger = screen.getByRole("button", { expanded: false });
    expect(trigger.textContent).toContain("exit 2");
    expect(trigger.textContent).not.toContain("·");
  });

  it("shows the stderr first line for a timed-out command", () => {
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
    expect(trigger.textContent).toContain("Timed out after 120s");
    expect(trigger.textContent).not.toContain("sleep 120");
  });

  it("prefers stdout first line on exit 0, falls back to 'exit 0'", () => {
    const { rerender } = render(
      <GenericTool
        part={makePart({
          state: "output-available",
          output: {
            exit_code: 0,
            stdout: "Hello, world!\nmore lines below\n",
            stderr: "",
          },
        })}
      />,
    );
    const trigger1 = screen.getByRole("button", { expanded: false });
    expect(trigger1.textContent).toContain("Hello, world!");
    expect(trigger1.textContent).not.toContain("more lines below");

    rerender(
      <GenericTool
        part={makePart({
          state: "output-available",
          output: { exit_code: 0, stdout: "", stderr: "" },
        })}
      />,
    );
    const trigger2 = screen.getByRole("button", { expanded: false });
    expect(trigger2.textContent).toContain("exit 0");
  });
});
