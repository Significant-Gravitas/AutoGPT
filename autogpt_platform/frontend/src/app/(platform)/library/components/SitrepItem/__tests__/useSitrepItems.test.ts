import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { getTenantEntityKey } from "@/services/org-team/identity";
import { describe, expect, it } from "vitest";
import { groupExecutionsByAgent } from "../useSitrepItems";

describe("groupExecutionsByAgent", () => {
  it("joins executions to the matching tenant copy of a duplicated graph", () => {
    const teamAAgent = {
      id: "library-a",
      graph_id: "shared-graph",
      organization_id: "org-1",
      team_id: "team-a",
    } as LibraryAgent;
    const teamBAgent = {
      id: "library-b",
      graph_id: "shared-graph",
      organization_id: "org-1",
      team_id: "team-b",
    } as LibraryAgent;
    const teamBExecution = {
      id: "run-b",
      graph_id: "shared-graph",
      organization_id: "org-1",
      team_id: "team-b",
    } as GraphExecutionMeta;
    const lookup = new Map([
      [getTenantEntityKey("shared-graph", "org-1", "team-a"), teamAAgent],
      [getTenantEntityKey("shared-graph", "org-1", "team-b"), teamBAgent],
    ]);

    const grouped = groupExecutionsByAgent([teamBExecution], lookup);

    expect(grouped.has(teamAAgent)).toBe(false);
    expect(grouped.get(teamBAgent)).toEqual([teamBExecution]);
  });
});
