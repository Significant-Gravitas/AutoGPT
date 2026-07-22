import { describe, expect, it } from "vitest";

import { getLibraryAgentBuilderHref } from "./helpers";

describe("getLibraryAgentBuilderHref", () => {
  it("builds a versionless builder URL for library agent edit actions", () => {
    expect(getLibraryAgentBuilderHref("graph-123")).toBe(
      "/build?flowID=graph-123",
    );
  });
});
