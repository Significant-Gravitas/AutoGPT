import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { toastIfPlanRequired } from "../helpers";
import { paywallDetail, TEAM_ORG_PLAN_REQUIRED } from "../paywall-detail";
import {
  PaywallToastAction,
  PaywallToastDescription,
  PlanRequiredDescription,
} from "../PaywallToast";

const toastSpy = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast: toastSpy,
}));

const billing = vi.hoisted(() => ({ enabled: true }));
vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => billing.enabled,
}));

const PERSONAL_ORG = {
  id: "org-personal",
  name: "Jane's Org",
  slug: "jane",
  avatarUrl: null,
  isPersonal: true,
  memberCount: 1,
};
const TEAM_ORG = {
  id: "org-team",
  name: "Acme Inc",
  slug: "acme",
  avatarUrl: null,
  isPersonal: false,
  memberCount: 4,
};

// What the backend says in a team organization without a plan
// (``paywall_message`` in backend/data/entitlement.py).
const ORG_NEEDS_A_PLAN =
  "This organization needs an active plan to use this feature. Ask its owner to contact AutoGPT about a team plan.";

function workIn(org: typeof PERSONAL_ORG) {
  useOrgTeamStore.setState({
    activeOrgID: org.id,
    orgs: [PERSONAL_ORG, TEAM_ORG],
    isLoaded: true,
  });
}

function refusal(detail: string | null) {
  return new ApiError(detail ?? "Payment Required", 402, {
    message: "Failed to process POST /api/integrations",
    detail,
  });
}

describe("the plan-required message", () => {
  beforeEach(() => {
    billing.enabled = true;
    toastSpy.mockClear();
  });

  afterEach(() => {
    useOrgTeamStore.setState({ activeOrgID: null, orgs: [], isLoaded: false });
  });

  describe("in a personal organization", () => {
    beforeEach(() => workIn(PERSONAL_ORG));

    it("asks the person to choose a plan and links to the plans", () => {
      render(
        <>
          <PlanRequiredDescription detail={ORG_NEEDS_A_PLAN} />
          <PaywallToastAction />
        </>,
      );

      expect(screen.getByText("Choose one to continue.")).toBeDefined();
      expect(
        screen.getByRole("link", { name: "See plans" }).getAttribute("href"),
      ).toBe("/settings/billing");
    });

    it("keeps the send toast's own wording", () => {
      render(
        <PaywallToastDescription recovery="your message is back in the composer" />,
      );

      expect(
        screen.getByText(
          "Choose one to keep chatting — your message is back in the composer.",
        ),
      ).toBeDefined();
    });
  });

  describe("in a team organization", () => {
    beforeEach(() => workIn(TEAM_ORG));

    it("says what the backend said, with no personal upgrade", () => {
      render(
        <>
          <PlanRequiredDescription detail={ORG_NEEDS_A_PLAN} />
          <PaywallToastAction />
        </>,
      );

      expect(screen.getByText(ORG_NEEDS_A_PLAN)).toBeDefined();
      expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
      expect(screen.queryByText("Choose one to continue.")).toBeNull();
    });

    it("says an organization owner can set up a team plan when the backend gave no reason", () => {
      render(<PlanRequiredDescription />);

      expect(screen.getByText(TEAM_ORG_PLAN_REQUIRED)).toBeDefined();
      expect(TEAM_ORG_PLAN_REQUIRED).toMatch(/owner can contact AutoGPT/);
    });

    it("puts the backend's reason before where the message went", () => {
      render(
        <PaywallToastDescription
          recovery="your message is back in the composer"
          detail={ORG_NEEDS_A_PLAN}
        />,
      );

      expect(
        screen.getByText(
          `${ORG_NEEDS_A_PLAN} Your message is back in the composer.`,
        ),
      ).toBeDefined();
    });

    it("even where this install doesn't bill", () => {
      billing.enabled = false;
      render(<PlanRequiredDescription detail={ORG_NEEDS_A_PLAN} />);

      expect(screen.getByText(ORG_NEEDS_A_PLAN)).toBeDefined();
    });
  });

  it("hands the backend's reason to the toast", () => {
    workIn(TEAM_ORG);

    expect(
      toastIfPlanRequired(
        refusal(ORG_NEEDS_A_PLAN),
        "Connecting a service needs an active plan",
      ),
    ).toBe(true);

    const shown = toastSpy.mock.calls[0][0];
    expect(shown.title).toBe("Connecting a service needs an active plan");
    render(
      <>
        {shown.description}
        {shown.action}
      </>,
    );
    expect(screen.getByText(ORG_NEEDS_A_PLAN)).toBeDefined();
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
  });
});

describe("paywallDetail", () => {
  it("reads an API error's detail", () => {
    expect(paywallDetail(refusal(ORG_NEEDS_A_PLAN))).toBe(ORG_NEEDS_A_PLAN);
  });

  it("reads the JSON body a refused chat send carries", () => {
    const sent = new Error(JSON.stringify({ detail: ORG_NEEDS_A_PLAN }));
    expect(paywallDetail(sent)).toBe(ORG_NEEDS_A_PLAN);
  });

  it.each([
    ["no detail", refusal(null)],
    ["a blank detail", refusal("  ")],
    ["plain text", new Error("Payment Required")],
    ["not an error", "402"],
  ])("gives nothing for %s", (_case, error) => {
    expect(paywallDetail(error)).toBeUndefined();
  });
});
