import { describe, expect, it } from "vitest";
import { normalizePartForRenderer } from "../components/MessageList";

describe("normalizePartForRenderer", () => {
  it("rewrites dynamic-tool parts to tool-<name> for MessagePartRenderer", () => {
    const part = {
      type: "dynamic-tool",
      toolName: "edit_agent",
      toolCallId: "tc-1",
      state: "output-available",
    };
    const out = normalizePartForRenderer(part);
    expect(out.type).toBe("tool-edit_agent");
    expect((out as unknown as { toolCallId: string }).toolCallId).toBe("tc-1");
  });

  it("leaves other part types untouched", () => {
    const part = { type: "text", text: "hello" };
    const out = normalizePartForRenderer(part);
    expect(out.type).toBe("text");
    expect((out as unknown as { text: string }).text).toBe("hello");
  });

  it("is safe on parts without a toolName field", () => {
    const part = { type: "dynamic-tool" };
    // Without toolName the runtime guard falls through — part passes through unchanged.
    const out = normalizePartForRenderer(part);
    expect(out.type).toBe("dynamic-tool");
  });

  it("ignores null and primitive inputs without throwing", () => {
    expect(() => normalizePartForRenderer(null)).not.toThrow();
    expect(() => normalizePartForRenderer(undefined)).not.toThrow();
    expect(() => normalizePartForRenderer("text")).not.toThrow();
  });

  it("renames run_agent dynamic tool parts", () => {
    const part = {
      type: "dynamic-tool",
      toolName: "run_agent",
      toolCallId: "tc-run",
      state: "output-available",
      output: { execution_id: "exec-1" },
    };
    const out = normalizePartForRenderer(part);
    expect(out.type).toBe("tool-run_agent");
  });
});
