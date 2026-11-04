import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { describe, expect, it } from "vitest";

import { ActivationPaymentMethod } from "../ActivationPaymentMethod";

const PORTAL_URL = "https://billing.stripe.com/p/session/test";

const TEAM_ORG = {
  id: "org-team",
  name: "Acme Inc",
  slug: "acme",
  avatarUrl: null,
  isPersonal: false,
  memberCount: 3,
};

const PERSONAL_ORG = {
  ...TEAM_ORG,
  id: "org-personal",
  name: "Jane's Org",
  slug: "jane",
  isPersonal: true,
  memberCount: 1,
};

function workIn(org: typeof TEAM_ORG) {
  useOrgTeamStore.setState({
    activeOrgID: org.id,
    activeTeamID: null,
    orgs: [PERSONAL_ORG, TEAM_ORG],
    teams: [],
    isLoaded: true,
  });
}

// The payment method route, as the backend answers it for a plain member of
// the team organization: refused there (no MANAGE_BILLING), served for the
// caller's own account. Records the X-Org-Id of each call ("none" without).
function mockPaymentMethodRoute() {
  const orgs: string[] = [];
  server.use(
    http.get("*/api/credits/manage", ({ request }) => {
      const org = request.headers.get("X-Org-Id") ?? "none";
      orgs.push(org);
      return org === TEAM_ORG.id
        ? HttpResponse.json(
            { detail: "Missing org permission: MANAGE_BILLING" },
            { status: 403 },
          )
        : HttpResponse.json({ url: PORTAL_URL });
    }),
  );
  return orgs;
}

function manageLink() {
  return screen.getByRole("link", {
    name: "Manage payment method in a new tab",
  });
}

describe("ActivationPaymentMethod", () => {
  it("in a team organization, opens the person's own payment method", async () => {
    workIn(TEAM_ORG);
    const orgs = mockPaymentMethodRoute();

    render(<ActivationPaymentMethod />);

    await waitFor(() =>
      expect(manageLink().getAttribute("href")).toBe(PORTAL_URL),
    );
    // A personal subscription belongs to the person's personal organization,
    // which the backend reaches when the request names no organization.
    expect(new Set(orgs)).toEqual(new Set(["none"]));
  });

  it("in a personal organization, asks as every request does", async () => {
    workIn(PERSONAL_ORG);
    const orgs = mockPaymentMethodRoute();

    render(<ActivationPaymentMethod />);

    await waitFor(() =>
      expect(manageLink().getAttribute("href")).toBe(PORTAL_URL),
    );
    expect(new Set(orgs)).toEqual(new Set([PERSONAL_ORG.id]));
  });
});
