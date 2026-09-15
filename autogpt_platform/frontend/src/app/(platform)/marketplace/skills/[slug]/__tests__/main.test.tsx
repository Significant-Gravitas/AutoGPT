import {
  getGetV2GetMarketplaceSkillMockHandler200,
  getPostV2InstallMarketplaceSkillMockHandler200,
  getGetV2GetMarketplaceSkillMockHandler404,
  getPostV2InstallMarketplaceSkillMockHandler401,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { getGetV1ListCredentialsMockHandler200 } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { MarketplaceSkillDetails } from "@/app/api/__generated__/models/marketplaceSkillDetails";
import { server } from "@/mocks/mock-server";
import { HttpResponse, http } from "msw";
import {
  configure,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { SkillPage } from "../components/SkillPage";

// These pages wait on several queries before anything renders, and CI is
// slower than a dev machine; the testing-library default is one second.
configure({ asyncUtilTimeout: 10000 });

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockNotFound = vi.hoisted(() => vi.fn());
const skillsHubFlag = vi.hoisted(() => ({ enabled: true, ready: true }));

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
  notFound: mockNotFound,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useFlagStatus: () => skillsHubFlag };
});

const outreach: MarketplaceSkillDetails = {
  slug: "outreach-playbook",
  name: "Outreach playbook",
  description: "Run cold outreach that gets replies.",
  categories: ["sales"],
  required_providers: ["google"],
  install_count: 3,
  creator: null,
  creator_avatar: null,
  skill_listing_version_id: "version-1",
  body: [
    "# Outreach playbook",
    "Four sentences, no more.",
    "## Before writing anything",
    "### Research",
    "Look for a **trigger** and record it as `trigger_note`.",
    "- No greeting beyond their first name",
    "1. The trigger, stated as a fact",
  ].join("\n\n"),
  triggers: ["cold email"],
  updated_at: new Date("2026-09-07T00:00:00Z"),
};

const googleCredential = {
  id: "cred-1",
  provider: "google",
  type: "oauth2",
  title: "Google",
} as CredentialsMetaResponse;

/** The user's own skills, which is where "Added" comes from. Mutable so a
 *  test can let the install land the way the app does. */
const userSkills: Array<{ name: string; description: string }> = [];

function userSkillsHandler() {
  return http.get("/api/proxy/api/skills", () => HttpResponse.json(userSkills));
}

function renderPage(credentials: CredentialsMetaResponse[]) {
  server.use(
    getGetV2GetMarketplaceSkillMockHandler200(outreach),
    getPostV2InstallMarketplaceSkillMockHandler200({
      name: "outreach-playbook",
      required_providers: ["google"],
    }),
    getGetV1ListCredentialsMockHandler200(credentials),
    userSkillsHandler(),
  );
  render(<SkillPage slug="outreach-playbook" />);
}

describe("Marketplace skill page", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    mockNotFound.mockClear();
    skillsHubFlag.enabled = true;
    skillsHubFlag.ready = true;
    userSkills.length = 0;
    server.use(userSkillsHandler());
  });

  test("is not found when the skills-hub flag is off", async () => {
    skillsHubFlag.enabled = false;
    renderPage([]);

    // A bookmarked URL reaches this page without passing the marketplace
    // shelf, so the shelf being hidden is not what keeps the feature dark.
    expect(mockNotFound).toHaveBeenCalled();
    expect(screen.queryByTestId("skill-install-button")).toBeNull();
  });

  test("shows the instructions experts will follow", async () => {
    renderPage([]);

    expect(
      await screen.findByText("Run cold outreach that gets replies."),
    ).toBeDefined();
    // The SKILL.md renders as prose, not as a wall of raw markdown: its own
    // "# Outreach playbook" becomes a heading rather than literal text.
    expect(await screen.findByText(/Four sentences, no more/)).toBeDefined();
    // The body's own leading "# Outreach playbook" is dropped, so the title
    // is the page's <h1> and nothing repeats it.
    expect(
      screen.getAllByRole("heading", { name: "Outreach playbook" }),
    ).toHaveLength(1);
  });

  test("installs in one click and offers the connect step afterwards", async () => {
    renderPage([]);

    const button = await screen.findByTestId(
      "skill-install-button",
      undefined,
      {
        timeout: 10000,
      },
    );
    server.use(
      http.get("/api/proxy/api/skills", () =>
        HttpResponse.json([
          { name: "outreach-playbook", description: "Installed copy" },
        ]),
      ),
    );
    await userEvent.click(button);

    expect(await screen.findByText("Installed")).toBeDefined();
    const connectStep = await screen.findByTestId("skill-connect-step");
    // The unconnected case is the normal path, so it must not read as a
    // failure: no error/warning wording, and the install already succeeded.
    expect(connectStep.textContent).toMatch(/Connect when you first need it/);
    expect(connectStep.textContent).not.toMatch(/error|required|must|cannot/i);
  });

  test("offers no connect step when the integration is already connected", async () => {
    renderPage([googleCredential]);

    const button = await screen.findByTestId(
      "skill-install-button",
      undefined,
      {
        timeout: 10000,
      },
    );
    server.use(
      http.get("/api/proxy/api/skills", () =>
        HttpResponse.json([
          { name: "outreach-playbook", description: "Installed copy" },
        ]),
      ),
    );
    await userEvent.click(button);

    expect(await screen.findByText("Installed")).toBeDefined();
    expect(screen.queryByTestId("skill-connect-step")).toBeNull();
  });

  test("sends a signed-out visitor to sign up, and back here afterwards", async () => {
    mockUseAuth.mockReturnValue({ user: null, isLoggedIn: false });
    renderPage([]);

    const cta = await screen.findByRole("link", { name: "Install skill" });
    expect(cta.getAttribute("href")).toBe(
      "/signup?next=%2Fmarketplace%2Fskills%2Foutreach-playbook",
    );
    expect(screen.queryByTestId("skill-install-button")).toBeNull();
  });

  test("marks a skill the user already has without them clicking", async () => {
    userSkills.push({ name: "outreach-playbook", description: "Installed" });
    renderPage([]);

    expect(
      await screen.findByText("Installed", undefined, {
        timeout: 10000,
      }),
    ).toBeDefined();
    expect(screen.queryByTestId("skill-install-button")).toBeNull();
  });

  test("renders every part of the SKILL.md, not just its paragraphs", async () => {
    renderPage([]);

    expect(await screen.findByText("Before writing anything")).toBeDefined();
    expect(await screen.findByText("Research")).toBeDefined();
    expect(await screen.findByText("trigger")).toBeDefined();
    expect(await screen.findByText("trigger_note")).toBeDefined();
    expect(
      await screen.findByText("No greeting beyond their first name"),
    ).toBeDefined();
    expect(
      await screen.findByText("The trigger, stated as a fact"),
    ).toBeDefined();
  });

  test("is a 404 when the listing is gone, not an error card", async () => {
    server.use(getGetV2GetMarketplaceSkillMockHandler404());
    render(<SkillPage slug="outreach-playbook" />);

    await waitFor(() => expect(mockNotFound).toHaveBeenCalled());
  });

  test("keeps the panel usable when the install fails", async () => {
    renderPage([]);
    await screen.findByTestId("skill-install-button", undefined, {
      timeout: 10000,
    });
    // A rejected install must not leave an unhandled rejection or a stuck
    // spinner: the button comes back and no success state is claimed.
    server.use(getPostV2InstallMarketplaceSkillMockHandler401());

    await userEvent.click(screen.getByTestId("skill-install-button"));

    expect(await screen.findByTestId("skill-install-button")).toBeDefined();
    expect(screen.queryByText("Installed")).toBeNull();
    expect(screen.queryByTestId("skill-connect-step")).toBeNull();
  });

  test("shows no connect step before the skill is installed", async () => {
    renderPage([]);

    await screen.findByTestId("skill-install-button", undefined, {
      timeout: 10000,
    });

    expect(screen.queryByTestId("skill-connect-step")).toBeNull();
  });
});
