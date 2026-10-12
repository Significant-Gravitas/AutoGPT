import { describe, expect, it } from "vitest";
import type { ApprovalItem } from "../../ApprovalQueue/helpers";
import type { ExpertIdentity } from "../../../useExpertMap";
import { capLine, editedReview, readHandoff } from "../helpers";

function item(overrides: Partial<ApprovalItem> = {}): ApprovalItem {
  return {
    args: { expert_id: "exp-alex", prompt: "Draft the PRD\nwith criteria" },
    spend: null,
    ...overrides,
  } as ApprovalItem;
}

const ROSTER = new Map<string, ExpertIdentity>([
  [
    "exp-alex",
    {
      id: "exp-alex",
      name: "Alex",
      role: "Product Manager",
      avatarUrl: "/alex.png",
      isArchived: false,
      readOnlyReason: null,
    },
  ],
]);

describe("readHandoff", () => {
  it("reads the server's handoff block first", () => {
    const facts = readHandoff(
      item(),
      {
        handoff: {
          expert: {
            id: "exp-bea",
            name: "Bea",
            role: "Ops",
            avatar_url: "/b.png",
          },
          title: "Onboarding revamp PRD",
          brief: "The real brief",
          why: "Bea owns onboarding",
          expected_back: "PRD draft as a file",
          by: "Today 6:00 pm",
        },
      },
      ROSTER,
    );
    expect(facts).toMatchObject({
      expert: { id: "exp-bea", name: "Bea", role: "Ops", avatarUrl: "/b.png" },
      title: "Onboarding revamp PRD",
      brief: "The real brief",
      why: "Bea owns onboarding",
      expectedBack: "PRD draft as a file",
      by: "Today 6:00 pm",
    });
  });

  it("accepts flat expert fields on the block", () => {
    const facts = readHandoff(
      item(),
      {
        handoff: { expert_name: "Cy", expert_id: "exp-cy", expert_role: "QA" },
      },
      ROSTER,
    );
    expect(facts.expert).toMatchObject({
      id: "exp-cy",
      name: "Cy",
      role: "QA",
    });
  });

  it("falls back to the call's args and the roster", () => {
    const facts = readHandoff(item(), {}, ROSTER);
    expect(facts).toMatchObject({
      expert: { name: "Alex", role: "Product Manager" },
      title: "Draft the PRD",
      brief: "Draft the PRD\nwith criteria",
      why: null,
      expectedBack: "A report in this chat",
      by: null,
    });
  });

  it("clips a long title and names an unknown expert generically", () => {
    const facts = readHandoff(
      item({
        args: {
          expert_id: "5f0c1d2e-aaaa-bbbb-cccc-1234567890ab",
          prompt: "y".repeat(90),
        },
      }),
      null,
      ROSTER,
    );
    expect(facts.title).toHaveLength(60);
    expect(facts.expert.name).toBe("your expert");
  });
});

describe("capLine / editedReview", () => {
  it("says what the chat spent of its ceiling", () => {
    expect(capLine(item())).toBeNull();
    expect(
      capLine(
        item({
          spend: { estimate: 1, spent: 2_000_000, ceiling: 5_000_000, unit: 1 },
        }),
      ),
    ).toBe("$2.00 of $5.00");
  });

  it("sends edits on top of the call's args, or nothing", () => {
    expect(editedReview(item(), {})).toEqual({});
    expect(editedReview(item(), { expert_id: "exp-bea" })).toEqual({
      reviewed_data: {
        expert_id: "exp-bea",
        prompt: "Draft the PRD\nwith criteria",
      },
    });
  });
});
