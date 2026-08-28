import { describe, expect, it } from "vitest";
import {
  founderSafeArtifactName,
  founderSafeMarkdown,
  founderSafeText,
} from "./founder-safe-text";

describe("founderSafeText", () => {
  it("removes paths, ids, JSON, shell output, and internal context", () => {
    const value = [
      "Prepared /tmp/private/report.json",
      "tool_call_id=call-1 graph_id=graph-1",
      '{"query":"secret search payload"}',
      "$ cat /tmp/private/report.json",
      "<expert_identity>internal instructions</expert_identity>",
      "<project_context>private project decisions</project_context>",
    ].join("\n");

    const safe = founderSafeText(value, "Safe update");

    expect(safe).toContain("Prepared a workspace file");
    expect(safe).not.toMatch(/\/tmp|tool_call_id|graph_id|secret|\$ cat/);
    expect(safe).not.toContain("internal instructions");
    expect(safe).not.toContain("private project decisions");
  });

  it("uses a fallback for structured payloads", () => {
    expect(founderSafeText('{"query":"secret"}', "Safe update")).toBe(
      "Safe update",
    );
  });

  it("shows only the public artifact name", () => {
    expect(founderSafeArtifactName("/sessions/secret/launch-plan.md")).toBe(
      "launch-plan.md",
    );
  });

  it("preserves useful markdown structure while removing internal details", () => {
    const safe = founderSafeMarkdown(
      "## Delivered\n\n- Report: /tmp/private/report.md\n- Next step",
      "Safe update",
    );

    expect(safe).toBe(
      "## Delivered\n\n- Report: a workspace file\n- Next step",
    );
  });

  it("preserves exact workspace artifact links", () => {
    const uri =
      "workspace://550e8400-e29b-41d4-a716-446655440000#text/markdown";

    expect(
      founderSafeMarkdown(`[Launch plan](${uri})`, "Safe update"),
    ).toContain(uri);
  });
});
