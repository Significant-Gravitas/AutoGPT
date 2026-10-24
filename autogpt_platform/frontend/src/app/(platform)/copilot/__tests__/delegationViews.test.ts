import { describe, expect, it } from "vitest";
import type { LiveDelegationStatus } from "../delegations";
import {
  delegationName,
  delegationTitle,
  formatCost,
  getDelegationStatusView,
  resolveDelegationExpert,
  threadHref,
} from "../delegationViews";
import { getStatusLineText } from "../components/DelegationStatusLine/helpers";
import { makeDelegation } from "./delegationFixture";

const ROSTER = new Map([
  [
    "exp-alex",
    {
      name: "Alex",
      role: "Product Manager",
      avatarUrl: "/a.png",
      color: "#abc",
    },
  ],
]);

describe("resolveDelegationExpert", () => {
  it("prefers the run's own expert, then the roster, then a typed name", () => {
    const named = makeDelegation({
      expert: {
        id: "x",
        name: "Bea",
        role: null,
        avatarUrl: null,
        color: null,
      },
    });
    expect(delegationName(named, ROSTER)).toBe("Bea");
    expect(resolveDelegationExpert(makeDelegation(), ROSTER)).toMatchObject({
      name: "Alex",
      role: "Product Manager",
      avatarUrl: "/a.png",
    });
    expect(delegationName(makeDelegation({ expertId: "Devon" }))).toBe("Devon");
    expect(
      delegationName(
        makeDelegation({ expertId: "5f0c1d2e-aaaa-bbbb-cccc-1234567890ab" }),
      ),
    ).toBe("Your expert");
  });
});

describe("delegation view helpers", () => {
  it("formats cost, title and thread link", () => {
    expect(formatCost(null)).toBeNull();
    expect(formatCost(0.3)).toBe("$0.30");
    expect(delegationTitle(makeDelegation({ prompt: null }), "Alex")).toBe(
      "Task for Alex",
    );
    expect(
      delegationTitle(makeDelegation({ prompt: "x".repeat(100) })),
    ).toHaveLength(78);
    expect(threadHref(makeDelegation({ link: "/l" }))).toBe("/l");
    expect(threadHref(makeDelegation())).toBe("/copilot?sessionId=sub-1");
    expect(threadHref(makeDelegation({ subSessionId: null }))).toBeNull();
  });

  it.each<[LiveDelegationStatus, string]>([
    ["proposed", "Waiting for you"],
    ["needs-input", "Needs you"],
    ["queued", "Queued"],
    ["running", "Working"],
    ["completed", "Done"],
    ["transferred", "Handed over"],
    ["failed", "Failed"],
    ["cancelled", "Cancelled"],
    ["unknown", "Unclear"],
  ])("names %s %j", (status, label) => {
    expect(getDelegationStatusView(status).label).toBe(label);
  });
});

describe("getStatusLineText", () => {
  const base = {
    name: "Alex",
    elapsedSeconds: 250,
    question: null,
    resumed: false,
  };
  const d = makeDelegation({ costUsd: 0.21, error: "budget cap" });

  it.each<[LiveDelegationStatus, string, string | null]>([
    ["proposed", "Alex is waiting for your approval", null],
    ["running", "1 expert working", "Alex · 4m 10s · $0.21"],
    ["transferred", "Handed over to Alex", null],
    ["failed", "Alex stopped", "budget cap · 4m 10s · $0.21"],
    ["cancelled", "You stopped Alex", null],
    [
      "unknown",
      "Alex · status unclear",
      "Live updates stopped; open the thread to check.",
    ],
  ])("words %s", (status, headline, detail) => {
    expect(getStatusLineText(d, { ...base, status })).toEqual({
      headline,
      detail,
    });
  });

  it("says resumed after the user answered, and names a question", () => {
    expect(
      getStatusLineText(d, { ...base, status: "running", resumed: true })
        .detail,
    ).toBe("Alex · resumed · 4m 10s · $0.21");
    expect(
      getStatusLineText(d, { ...base, status: "needs-input", question: "Q4?" }),
    ).toEqual({ headline: "Alex needs you", detail: "Q4?" });
    expect(
      getStatusLineText(d, {
        ...base,
        name: "Your expert",
        status: "cancelled",
      }).headline,
    ).toBe("You stopped your expert");
  });
});
