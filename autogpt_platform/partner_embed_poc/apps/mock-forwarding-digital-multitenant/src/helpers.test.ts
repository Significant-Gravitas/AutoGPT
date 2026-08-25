import { describe, expect, it } from "vitest";

import {
  agentPermissionMessageFor,
  assistantNoticeFor,
  documentsForJobs,
  suggestedPromptsFor,
} from "./helpers";

describe("assistantNoticeFor", () => {
  it("describes saved agents without exposing their internal ID", () => {
    expect(assistantNoticeFor("/library/agents/agent-123")).toBe(
      "Saved agent is ready in this tenant's automation library.",
    );
  });

  it("uses a generic label for other host-owned resources", () => {
    expect(assistantNoticeFor("/library/templates/template-123")).toBe(
      "Saved resource is ready in this tenant's automation library.",
    );
  });

  it("builds document rows only from the active tenant's jobs", () => {
    const documents = documentsForJobs([
      { reference: "HBR-2208", status: "Rail slot pending" },
      { reference: "HBR-2231", status: "On schedule" },
    ]);

    expect(documents.map((document) => document.name)).toEqual([
      "Arrival notice · HBR-2208",
      "Bill of lading · HBR-2231",
    ]);
    expect(documents.map((document) => document.state)).toEqual([
      "Needs review",
      "Verified",
    ]);
  });
});

describe("suggestedPromptsFor", () => {
  it("offers restricted operators only actions their role can perform", () => {
    const capabilities = ["jobs.read", "documents.read"];

    expect(suggestedPromptsFor("documents", capabilities)).toEqual([
      "Find jobs with missing documents and produce an exception summary.",
    ]);
    expect(suggestedPromptsFor("automations", capabilities)).toEqual([
      "Turn the current shipment exceptions into a repeatable checklist a manager could automate.",
      "Review this tenant's document gaps and outline the safest manual follow-up workflow.",
    ]);
  });

  it("offers the full lifecycle only when every matching capability is enabled", () => {
    const prompts = suggestedPromptsFor("automations", [
      "agents.create",
      "agents.run",
      "agents.schedule",
      "autogpt:block:calculator",
    ]);

    expect(prompts).toHaveLength(3);
    expect(prompts[0]).toContain("Create and save");
    expect(prompts[1]).toContain("Run the saved");
    expect(prompts[2]).toContain("schedule the saved");
  });

  it("does not suggest creation without an enabled block", () => {
    expect(suggestedPromptsFor("automations", ["agents.create"])).toEqual([
      "Explain what this role can access and which additional capability an administrator would need to grant.",
    ]);
  });
  describe("agentPermissionMessageFor", () => {
    it("describes partial agent controls without claiming broader access", () => {
      expect(agentPermissionMessageFor(["agents.run"])).toBe(
        "Agent controls enabled for this role: run. Other actions remain unavailable.",
      );
      expect(
        agentPermissionMessageFor([
          "agents.create",
          "agents.run",
          "agents.schedule",
        ]),
      ).toBe(
        "Agent controls enabled for this role: run and schedule. Other actions remain unavailable.",
      );
    });
  });
});
