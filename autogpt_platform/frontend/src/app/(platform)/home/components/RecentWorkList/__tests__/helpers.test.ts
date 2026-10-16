import type { HomeRecentWork } from "@/app/api/__generated__/models/homeRecentWork";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { expect, it } from "vitest";
import { buildRecentWorkLines } from "../helpers";

function file(
  id: string,
  overrides: Partial<WorkspaceFileItem> = {},
): WorkspaceFileItem {
  return {
    id,
    name: `${id}.txt`,
    path: `/sessions/chat/${id}.txt`,
    mime_type: "text/plain",
    size_bytes: 3,
    origin: "generated",
    created_at: "2026-10-16T12:00:00Z",
    ...overrides,
  };
}

it("shows visible chat files and links them to their chat", () => {
  const lines = buildRecentWorkLines(
    undefined,
    [
      file("report", { expert_id: "expert" }),
      file("hidden", { name: "agent.json" }),
      file("tool", { metadata: { purpose: "tool-output" } }),
      file("workflow", { path: "/runs/run/report.txt" }),
      file("invalid", { created_at: "bad date" }),
    ],
    new Map([
      [
        "expert",
        {
          id: "expert",
          name: "Maria",
          role: "Marketing",
          avatarUrl: null,
          isArchived: false,
          readOnlyReason: null,
        },
      ],
    ]),
  );
  expect(lines).toHaveLength(1);
  expect(lines[0]).toMatchObject({
    actor: { kind: "expert", name: "Maria" },
    verb: "made",
    object: "report.txt",
    link: "/home?sessionId=chat",
  });
});

it("combines run outcomes, schedule changes, file deletions and repeated service use", () => {
  const at = new Date("2026-10-16T12:00:00Z");
  const recent: HomeRecentWork = {
    groups: [
      {
        actor: { kind: "autopilot", name: "Otto" },
        latest_at: at,
        runs: [
          {
            id: "run",
            status: "failed",
            title: "Report",
            agent_name: "Report",
            summary: "Failed",
            occurred_at: at,
            link: "/runs/run",
          },
        ],
        items: [
          {
            id: "s",
            category: "schedule",
            event_type: "schedule.paused",
            title: "Daily report",
            occurred_at: at,
          },
          {
            id: "f",
            category: "file",
            event_type: "file.deleted",
            title: "Draft.txt",
            occurred_at: at,
          },
          {
            id: "created",
            category: "file",
            event_type: "file.created",
            title: "Duplicate",
            occurred_at: at,
          },
          {
            id: "i1",
            category: "integration",
            event_type: "tool.used",
            title: "Search",
            provider: "github",
            occurred_at: at,
          },
          {
            id: "i2",
            category: "integration",
            event_type: "tool.used",
            title: "Search",
            provider: "github",
            occurred_at: at,
          },
        ],
      },
    ],
  };
  const lines = buildRecentWorkLines(recent, [], new Map());
  expect(lines).toHaveLength(4);
  expect(lines[0]).toMatchObject({
    verb: "couldn't finish",
    failed: true,
    link: "/runs/run",
  });
  expect(lines[1].verb).toBe("paused");
  expect(lines[2]).toMatchObject({ verb: "deleted", object: "Draft.txt" });
  expect(lines[3]).toMatchObject({ verb: "used GitHub", suffix: "2 times" });
});

it("keeps the five newest results and handles an empty feed", () => {
  const files = Array.from({ length: 8 }, (_, i) =>
    file(String(i), { created_at: `2026-10-${10 + i}T12:00:00Z` }),
  );
  expect(
    buildRecentWorkLines(undefined, files, new Map()).map((line) => line.id),
  ).toEqual(["file:7", "file:6", "file:5", "file:4", "file:3"]);
  expect(buildRecentWorkLines(undefined, [], new Map())).toEqual([]);
});
