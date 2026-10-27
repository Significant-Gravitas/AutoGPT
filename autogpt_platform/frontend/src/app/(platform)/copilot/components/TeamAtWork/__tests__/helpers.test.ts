import type { ExpertFirstJob } from "@/app/api/__generated__/models/expertFirstJob";
import { describe, expect, it } from "vitest";
import {
  getFirstJobDetail,
  getFirstJobHref,
  getStatusLabel,
  shouldPoll,
} from "../helpers";

function job(overrides: Partial<ExpertFirstJob> = {}): ExpertFirstJob {
  return {
    expert_id: "expert-1",
    name: "Ava",
    role: "Inbox manager",
    avatar_url: null,
    title: "Sort this week's inbox",
    description: "Label every unread email and flag what needs you.",
    status: "open",
    detail: "Not started yet",
    providers: [],
    session_id: null,
    ...overrides,
  };
}

describe("getFirstJobHref", () => {
  it("starts the expert's kickoff for a job nobody started", () => {
    expect(getFirstJobHref(job({ status: "open" }))).toBe(
      "/copilot?expertId=expert-1&kickoff=1",
    );
  });

  it.each(["working", "ready", "needs_you"] as const)(
    "opens the chat for a %s job with a session",
    (status) => {
      expect(getFirstJobHref(job({ status, session_id: "session-9" }))).toBe(
        "/copilot?sessionId=session-9",
      );
    },
  );

  it("falls back to the expert's page for a started job without a session", () => {
    expect(getFirstJobHref(job({ status: "working", session_id: null }))).toBe(
      "/team/expert-1",
    );
  });

  it("encodes the ids it puts in a link", () => {
    expect(getFirstJobHref(job({ expert_id: "a/b&c" }))).toBe(
      "/copilot?expertId=a%2Fb%26c&kickoff=1",
    );
    expect(getFirstJobHref(job({ status: "ready", session_id: "s?1#2" }))).toBe(
      "/copilot?sessionId=s%3F1%232",
    );
    expect(
      getFirstJobHref(job({ status: "installing", expert_id: "a/b" })),
    ).toBe("/team/a%2Fb");
  });

  it.each(["installing", "needs_setup"] as const)(
    "sends a %s job to the expert's page",
    (status) => {
      expect(getFirstJobHref(job({ status, session_id: "session-9" }))).toBe(
        "/team/expert-1",
      );
    },
  );
});

describe("getStatusLabel", () => {
  it.each([
    ["installing", "Installing"],
    ["needs_setup", "Needs setup"],
    ["needs_you", "Needs you"],
    ["working", "Working"],
    ["ready", "Ready for review"],
    ["open", "Start a chat"],
  ] as const)("labels %s as %s", (status, label) => {
    expect(getStatusLabel(status)).toBe(label);
  });
});

describe("shouldPoll", () => {
  it("polls while any expert is installing or working", () => {
    expect(
      shouldPoll([job({ status: "ready" }), job({ status: "working" })]),
    ).toBe(true);
    expect(shouldPoll([job({ status: "installing" })])).toBe(true);
  });

  it("stops once nothing is moving on its own", () => {
    expect(
      shouldPoll([
        job({ status: "open" }),
        job({ status: "ready" }),
        job({ status: "needs_you" }),
        job({ status: "needs_setup" }),
      ]),
    ).toBe(false);
    expect(shouldPoll([])).toBe(false);
  });
});

describe("getFirstJobDetail", () => {
  it("names the tools a job needs set up", () => {
    expect(
      getFirstJobDetail(
        job({
          status: "needs_setup",
          detail: "Needs google, notion",
          providers: ["google", "notion", "mcp:linear"],
        }),
      ),
    ).toBe("Needs Google, Notion, linear");
  });

  it("keeps the server's detail otherwise", () => {
    expect(
      getFirstJobDetail(
        job({ status: "needs_setup", detail: "Setup failed", providers: [] }),
      ),
    ).toBe("Setup failed");
    expect(getFirstJobDetail(job({ status: "open" }))).toBe("Not started yet");
  });
});
