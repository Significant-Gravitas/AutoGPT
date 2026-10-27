import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { describe, expect, it } from "vitest";
import {
  getKickoffLabel,
  getMentionedProviders,
  sortKickoffFirst,
} from "../helpers";

function task(id: string, extra: Partial<SuggestedTask> = {}): SuggestedTask {
  return { id, title: `Task ${id}`, prompt: `Do ${id}`, ...extra };
}

describe("getMentionedProviders", () => {
  it.each([
    ["No tools named here", []],
    ["Plan Monday's work with teams on a medium budget", []],
    ["Build a linear model and mark X on the chart", []],
    [
      "Summarize GITHUB updates in Slack, then GitHub again",
      ["github", "slack"],
    ],
    ["Sync Microsoft Teams with Monday.com", ["teams", "monday"]],
    ["Read Gmail and Google Calendar", ["google"]],
    ["Find Jira, Confluence, and Atlassian issues", ["atlassian"]],
    [
      "Send Slack, GitHub, Linear, and Notion updates",
      ["slack", "github", "linear"],
    ],
    ["Review githubish and slackbot tools", []],
  ])("finds providers in %s", (title, expected) => {
    expect(getMentionedProviders(title)).toEqual(expected);
  });
});

describe("sortKickoffFirst", () => {
  it("puts the drafts first and keeps the order within each group", () => {
    const tasks = [
      task("0"),
      task("1", { kickoff: true }),
      task("2"),
      task("3", { kickoff: true }),
    ];

    expect(sortKickoffFirst(tasks).map(({ id }) => id)).toEqual([
      "1",
      "3",
      "0",
      "2",
    ]);
  });

  it("leaves a list without drafts as it was", () => {
    const tasks = [task("0"), task("1", { kickoff: false }), task("2")];

    expect(sortKickoffFirst(tasks)).toEqual(tasks);
  });
});

describe("getKickoffLabel", () => {
  it("says Otto is on it while the draft is running", () => {
    expect(
      getKickoffLabel(task("0", { kickoff: true, kickoff_status: "working" })),
    ).toBe("Otto is working on it");
  });

  it("offers a finished draft for review", () => {
    expect(
      getKickoffLabel(task("0", { kickoff: true, kickoff_status: "ready" })),
    ).toBe("Draft · ready to review");
  });
});
