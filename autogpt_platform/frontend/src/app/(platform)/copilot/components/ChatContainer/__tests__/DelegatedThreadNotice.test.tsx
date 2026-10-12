import {
  getGetDelegationSettingsMockHandler200,
  getListDelegationsMockHandler200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import type { DelegationSummary } from "@/app/api/__generated__/models/delegationSummary";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import type { UIMessage } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DelegatedThreadNotice } from "../components/DelegatedThreadNotice/DelegatedThreadNotice";
import { getOpeningText } from "../components/DelegatedThreadNotice/helpers";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

const ALEX = { name: "Alex", avatarUrl: null };

function handoff(over: Partial<DelegationSummary> = {}): DelegationSummary {
  return {
    sub_session_id: "sub-1",
    parent_session_id: "parent-1",
    expert: { id: "exp-alex", name: "Alex", role: "Product Manager" },
    title: "Onboarding revamp PRD",
    brief: "Draft the PRD for the onboarding revamp.",
    status: "running",
    created_at: new Date(2026, 8, 28, 10, 41),
    finished_at: null,
    elapsed_seconds: 60,
    cost_usd: 0.12,
    files_count: 0,
    question: null,
    question_options: [],
    ...over,
  };
}

function withHandoffs(delegations: DelegationSummary[]) {
  server.use(
    getListDelegationsMockHandler200({
      delegations,
      total: delegations.length,
      summary: {
        working: 1,
        needs_you: 0,
        completed: 0,
        failed: 0,
        spent_today_usd: 0.12,
      },
    }),
    getGetDelegationSettingsMockHandler200({ per_delegation_cap_usd: 2 }),
  );
}

describe("DelegatedThreadNotice", () => {
  afterEach(cleanup);

  it("leads with Otto's hand-off: brief, cap used and the way back", async () => {
    withHandoffs([
      handoff(),
      handoff({ sub_session_id: "sub-2", title: "Other" }),
    ]);
    render(
      <DelegatedThreadNotice
        sentFrom={{ sessionId: "parent-1", expertId: null, expertName: null }}
        sessionId="sub-1"
        expert={ALEX}
        openingText="Delegated task from Otto (not the user): draft the PRD"
      />,
    );
    const notice = screen.getByTestId("delegated-thread-notice");

    expect(await screen.findByText("Working for Otto")).toBeDefined();
    expect(notice.textContent).toContain("Delegated by Otto · 10:41");
    expect(notice.textContent).toContain(
      "Part of “Onboarding revamp PRD” · you own the outcome",
    );
    expect(notice.textContent).toContain(
      "Draft the PRD for the onboarding revamp.",
    );
    expect(notice.textContent).toContain("A report in Otto's chat");
    expect(await screen.findByText("$2.00 cap · $0.12 used")).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: /back to otto's thread/i })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=parent-1");
  });

  it("says the work is done once the hand-off completes", async () => {
    withHandoffs([handoff({ status: "completed" })]);
    render(
      <DelegatedThreadNotice
        sentFrom={{ sessionId: "parent-1", expertId: null, expertName: null }}
        sessionId="sub-1"
        expert={ALEX}
        openingText={null}
      />,
    );

    expect(await screen.findByText("Done for Otto")).toBeDefined();
  });

  it("falls back to the opening message when Otto's chat has no record", async () => {
    withHandoffs([]);
    render(
      <DelegatedThreadNotice
        sentFrom={{
          sessionId: "parent-2",
          expertId: "exp-devon",
          expertName: "Devon",
        }}
        sessionId="sub-9"
        expert={ALEX}
        openingText="Estimate the onboarding revamp"
      />,
    );
    const notice = screen.getByTestId("delegated-thread-notice");

    expect(notice.textContent).toContain("Delegated by Devon");
    expect(notice.textContent).not.toContain("you own the outcome");
    expect(notice.textContent).toContain("Estimate the onboarding revamp");
    expect(
      screen
        .getByRole("link", { name: /back to devon's thread/i })
        .getAttribute("href"),
    ).toBe("/copilot?sessionId=parent-2");
  });

  it("reads the brief off the message that opened the thread", () => {
    const messages = [
      {
        id: "sub-1-seq-0",
        role: "user",
        parts: [{ type: "text", text: "Draft the PRD" }],
      },
      {
        id: "sub-1-seq-1",
        role: "assistant",
        parts: [{ type: "text", text: "On it." }],
      },
    ] as UIMessage[];
    expect(getOpeningText(messages)).toBe("Draft the PRD");
    expect(getOpeningText(messages.slice(1))).toBeNull();
  });
});
