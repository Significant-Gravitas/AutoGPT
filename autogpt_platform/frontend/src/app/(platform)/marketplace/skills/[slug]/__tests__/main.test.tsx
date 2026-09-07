import {
  getGetV2GetMarketplaceSkillMockHandler200,
  getPostV2InstallMarketplaceSkillMockHandler200,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { getGetV1ListCredentialsMockHandler200 } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { MarketplaceSkillDetails } from "@/app/api/__generated__/models/marketplaceSkillDetails";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { SkillPage } from "../components/SkillPage";

const mockUseAuth = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => "/marketplace/skills/outreach-playbook",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ slug: "outreach-playbook" }),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

const outreach: MarketplaceSkillDetails = {
  slug: "outreach-playbook",
  name: "Outreach playbook",
  description: "Run cold outreach that gets replies.",
  categories: ["sales"],
  required_providers: ["google"],
  is_verified: true,
  install_count: 3,
  creator: null,
  creator_avatar: null,
  skill_listing_version_id: "version-1",
  body: "# Outreach playbook\nFour sentences, no more.\n",
  triggers: ["cold email"],
  updated_at: new Date("2026-09-07T00:00:00Z"),
};

const googleCredential = {
  id: "cred-1",
  provider: "google",
  type: "oauth2",
  title: "Google",
} as CredentialsMetaResponse;

function renderPage(credentials: CredentialsMetaResponse[]) {
  server.use(
    getGetV2GetMarketplaceSkillMockHandler200(outreach),
    getPostV2InstallMarketplaceSkillMockHandler200({
      name: "outreach-playbook",
      required_providers: ["google"],
    }),
    getGetV1ListCredentialsMockHandler200(credentials),
  );
  render(<SkillPage slug="outreach-playbook" />);
}

describe("Marketplace skill page", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
  });

  test("shows the instructions the AutoPilot will follow", async () => {
    renderPage([]);

    expect(
      await screen.findByText("Run cold outreach that gets replies."),
    ).toBeDefined();
    // The SKILL.md renders as prose, not as a wall of raw markdown: its own
    // "# Outreach playbook" becomes a heading rather than literal text.
    expect(await screen.findByText(/Four sentences, no more/)).toBeDefined();
    expect(
      screen.getAllByRole("heading", { name: "Outreach playbook" }).length,
    ).toBeGreaterThan(0);
  });

  test("installs in one click and offers the connect step afterwards", async () => {
    renderPage([]);

    const button = await screen.findByTestId("skill-install-button");
    await userEvent.click(button);

    expect(await screen.findByTestId("skill-installed")).toBeDefined();
    const connectStep = await screen.findByTestId("skill-connect-step");
    // The unconnected case is the normal path, so it must not read as a
    // failure: no error/warning wording, and the install already succeeded.
    expect(connectStep.textContent).toMatch(/Connect when you first need it/);
    expect(connectStep.textContent).not.toMatch(/error|required|must|cannot/i);
  });

  test("offers no connect step when the integration is already connected", async () => {
    renderPage([googleCredential]);

    await userEvent.click(await screen.findByTestId("skill-install-button"));

    expect(await screen.findByTestId("skill-installed")).toBeDefined();
    expect(screen.queryByTestId("skill-connect-step")).toBeNull();
  });

  test("sends a signed-out visitor to log in rather than a dead button", async () => {
    mockUseAuth.mockReturnValue({ user: null, isLoggedIn: false });
    renderPage([]);

    const cta = await screen.findByRole("link", { name: "Add to AutoPilot" });
    expect(cta.getAttribute("href")).toBe("/login");
    expect(screen.queryByTestId("skill-install-button")).toBeNull();
  });

  test("shows no connect step before the skill is installed", async () => {
    renderPage([]);

    await screen.findByTestId("skill-install-button");

    expect(screen.queryByTestId("skill-connect-step")).toBeNull();
  });
});
