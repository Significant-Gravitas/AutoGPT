import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";

import { CopilotAuthError } from "../../../../../helpers";
import {
  ChatUnavailableError,
  SendRefusedError,
} from "../../../../../stream/turnRuntime";
import { ChatErrorBanner } from "../ChatErrorBanner";

const billing = vi.hoisted(() => ({ enabled: true }));
vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => billing.enabled,
}));

const NO_PLAN = new SendRefusedError(
  '{"detail":"A subscription is required to use this feature. Upgrade to continue."}',
  402,
);

// What the backend says in a team organization without a plan
// (``paywall_message`` in backend/data/entitlement.py).
const ORG_NEEDS_A_PLAN =
  "This organization needs an active plan to use this feature. Ask its owner to contact AutoGPT about a team plan.";
const NO_ORG_PLAN = new SendRefusedError(
  JSON.stringify({
    message: "Failed to process POST /api/chat/sessions/s-1/stream",
    detail: ORG_NEEDS_A_PLAN,
    hint: "Adjust the request and retry.",
  }),
  402,
);

function workInATeamOrg() {
  useOrgTeamStore.setState({
    activeOrgID: "org-team",
    orgs: [
      {
        id: "org-team",
        name: "Acme Inc",
        slug: "acme",
        avatarUrl: null,
        isPersonal: false,
        memberCount: 4,
      },
    ],
    isLoaded: true,
  });
}

afterEach(() => {
  window.history.replaceState(null, "", "/");
  billing.enabled = true;
  useOrgTeamStore.setState({ activeOrgID: null, orgs: [], isLoaded: false });
});

describe("ChatErrorBanner", () => {
  it("signed out: offers sign-in that returns to this chat", () => {
    window.history.replaceState(null, "", "/copilot?sessionId=s-1");
    render(<ChatErrorBanner error={new CopilotAuthError()} />);

    expect(
      screen.getByText("You're signed out. Sign in again to keep chatting."),
    ).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Sign in" }).getAttribute("href"),
    ).toBe(`/login?next=${encodeURIComponent("/copilot?sessionId=s-1")}`);
  });

  it("no plan: offers plans instead of a reload", () => {
    render(<ChatErrorBanner error={NO_PLAN} />);

    expect(
      screen.getByText(
        "Sending needs an active plan, so this message wasn't sent.",
      ),
    ).toBeDefined();
    expect(
      screen.getByRole("link", { name: "See plans" }).getAttribute("href"),
    ).toBe("/settings/billing");
    expect(screen.queryByText(/maintainer has been notified/)).toBeNull();
  });

  it("no plan where this install doesn't bill: no plans link, asks the operator", () => {
    billing.enabled = false;
    render(<ChatErrorBanner error={NO_PLAN} />);

    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
    expect(
      screen.getByText("Ask whoever runs this server to set one up."),
    ).toBeDefined();
  });

  it("no plan in a team organization: the backend's reason, and no personal plans", () => {
    workInATeamOrg();
    render(<ChatErrorBanner error={NO_ORG_PLAN} />);

    expect(screen.getByText("This message wasn't sent.")).toBeDefined();
    expect(screen.getByText(ORG_NEEDS_A_PLAN)).toBeDefined();
    expect(screen.queryByRole("link", { name: "See plans" })).toBeNull();
    expect(
      screen.queryByText(
        "Sending needs an active plan, so this message wasn't sent.",
      ),
    ).toBeNull();
  });

  it("a chat that is gone: offers a new chat", () => {
    render(<ChatErrorBanner error={new ChatUnavailableError()} />);

    expect(
      screen.getByText(
        "This chat no longer exists. Start a new chat to keep going.",
      ),
    ).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: "Start a new chat" })
        .getAttribute("href"),
    ).toBe("/copilot");
  });

  it("a refusal the app never provokes: the refusal, its response body folded away, and the notified line", () => {
    render(
      <ChatErrorBanner
        error={
          new SendRefusedError('{"detail":"advanced_tier_not_entitled"}', 403)
        }
      />,
    );

    const banner = screen.getByTestId("chat-error-banner");
    const details = banner.querySelector("details");
    expect(details?.open).toBe(false);
    expect(details?.querySelector("summary")?.textContent).toBe(
      "The request was refused. Reload the page and try again; if it keeps happening, contact support.",
    );
    expect(details?.querySelector("pre")?.textContent).toBe(
      '{"detail":"advanced_tier_not_entitled"}',
    );
    expect(
      Array.from(banner.querySelectorAll("p")).map((p) => p.textContent),
    ).toEqual(["The system maintainer has been notified of the issue"]);
  });

  it("a dropped connection: says to send again, with no action", () => {
    render(<ChatErrorBanner error={new TypeError("Failed to fetch")} />);

    expect(
      screen.getByText(
        "The connection dropped before your message was sent. Check your connection and send it again.",
      ),
    ).toBeDefined();
    expect(screen.queryByRole("link")).toBeNull();
    expect(
      screen.queryByText(
        "The system maintainer has been notified of the issue",
      ),
    ).toBeNull();
  });
});
