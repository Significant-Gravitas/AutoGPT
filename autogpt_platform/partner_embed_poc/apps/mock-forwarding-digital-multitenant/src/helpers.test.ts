import { describe, expect, it } from "vitest";

import { assistantNoticeFor, documentsForJobs } from "./helpers";

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
