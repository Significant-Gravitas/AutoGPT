import { describe, expect, it } from "vitest";
import { getMentionedProviders } from "../helpers";

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
