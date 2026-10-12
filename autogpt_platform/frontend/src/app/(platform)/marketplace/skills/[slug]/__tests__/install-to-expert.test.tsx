import { getListExpertsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListCredentialsMockHandler200 } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import {
  getGetV2GetMarketplaceSkillMockHandler200,
  getPostV2InstallMarketplaceSkillMockHandler200,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { MarketplaceSkillDetails } from "@/app/api/__generated__/models/marketplaceSkillDetails";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import {
  configure,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { SkillPage } from "../components/SkillPage";

configure({ asyncUtilTimeout: 10000 });

const mockUseAuth = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/marketplace/skills/outreach-playbook",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ slug: "outreach-playbook" }),
  notFound: vi.fn(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useFlagStatus: () => ({ enabled: true, ready: true }) };
});

const outreach: MarketplaceSkillDetails = {
  slug: "outreach-playbook",
  name: "outreach-playbook",
  title: "Outreach playbook",
  description: "Run cold outreach that gets replies.",
  categories: ["sales"],
  required_providers: [],
  install_count: 3,
  creator: null,
  creator_avatar: null,
  skill_listing_version_id: "version-1",
  body: "# Outreach playbook\n\nFour sentences, no more.",
  triggers: [],
  updated_at: new Date("2026-09-07T00:00:00Z"),
};

const maria: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: null,
  skills: [],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria.",
  voice_preferences: "Warm and direct.",
  boundaries: "Never invent customer evidence.",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

/** The query string the install landed on, or null when it never ran. */
let installTarget: string | null;

function renderPage(experts: Expert[]) {
  installTarget = null;
  server.use(
    getGetV2GetMarketplaceSkillMockHandler200(outreach),
    getPostV2InstallMarketplaceSkillMockHandler200(({ request }) => {
      installTarget = new URL(request.url).searchParams.get("expert_id") ?? "";
      return { name: "outreach-playbook", required_providers: [] };
    }),
    getGetV1ListCredentialsMockHandler200([]),
    getListExpertsMockHandler(experts),
    http.get("/api/proxy/api/skills", () => HttpResponse.json([])),
  );
  render(
    <>
      <SkillPage slug="outreach-playbook" />
      <Toaster />
    </>,
  );
}

describe("Installing a marketplace skill for an expert", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
  });

  test("the main half of the split button installs into your own skills", async () => {
    renderPage([maria]);

    await userEvent.click(await screen.findByTestId("skill-install-button"));

    // "" is the install landing in the caller's own skills: the request ran
    // and carried no expert, which null could not tell apart from no request.
    await waitFor(() => expect(installTarget).toBe(""));
  });

  test("the dropdown installs into the expert you pick, and says so", async () => {
    renderPage([maria]);
    await screen.findByTestId("skill-install-button");

    await userEvent.click(
      screen.getByRole("button", { name: "Install for an expert" }),
    );
    await userEvent.click(
      await screen.findByRole("menuitem", { name: /Maria/ }),
    );

    expect(await screen.findByText("Installed on Maria")).toBeDefined();
    expect(installTarget).toBe("expert-maria");
  });

  test("offers no dropdown when there is no expert to install onto", async () => {
    renderPage([]);

    await screen.findByTestId("skill-install-button");
    expect(
      screen.queryByRole("button", { name: "Install for an expert" }),
    ).toBeNull();
  });
});
