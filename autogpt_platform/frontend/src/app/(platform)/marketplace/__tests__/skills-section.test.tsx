import { getListExpertTemplatesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getListCopilotSkillsMockHandler200 } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import {
  getGetV2ListMarketplaceSkillsMockHandler200,
  getGetV2ListStoreAgentsMockHandler,
  getGetV2ListStoreCreatorsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import { server } from "@/mocks/mock-server";
import { HttpResponse, http } from "msw";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { MainMarkeplacePage } from "../components/MainMarketplacePage/MainMarketplacePage";

const mockUseAuth = vi.hoisted(() => vi.fn());
const flags = vi.hoisted(() => ({
  skillsHub: true,
  skillsHubReady: true,
  hireExperts: false,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
    push: vi.fn(),
    refresh: vi.fn(),
    replace: vi.fn(),
  }),
  usePathname: () => "/marketplace",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) => {
      if (flag === "skills-hub") return flags.skillsHub;
      if (flag === "hire-experts") return flags.hireExperts;
      return actual.useGetFlag(flag as never);
    },
    useFlagStatus: (flag: string) => {
      if (flag === "skills-hub")
        return { enabled: flags.skillsHub, ready: flags.skillsHubReady };
      return actual.useFlagStatus(flag as never);
    },
  };
});

const brandVoice: MarketplaceSkill = {
  slug: "brand-voice-guide",
  name: "brand-voice-guide",
  description: "Write in a consistent brand voice.",
  categories: ["content"],
  required_providers: [],
  install_count: 12,
  creator: null,
  creator_avatar: null,
};

const outreach: MarketplaceSkill = {
  ...brandVoice,
  slug: "outreach-playbook",
  name: "outreach-playbook",
  categories: ["sales"],
  required_providers: ["google"],
};

function listing(skills: MarketplaceSkill[], totalItems = skills.length) {
  return getGetV2ListMarketplaceSkillsMockHandler200({
    skills,
    pagination: {
      total_items: totalItems,
      total_pages: 1,
      current_page: 1,
      page_size: 4,
    },
  });
}

describe("Marketplace SkillsSection", () => {
  beforeEach(() => {
    flags.skillsHub = true;
    flags.skillsHubReady = true;
    flags.hireExperts = false;
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getGetV2ListStoreAgentsMockHandler(),
      getGetV2ListStoreCreatorsMockHandler(),
      getListExpertTemplatesMockHandler([]),
      getListCopilotSkillsMockHandler200([]),
    );
  });

  test("shows skills as their own shelf, linked to the skill page", async () => {
    server.use(listing([brandVoice]));

    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("AutoPilot Skills", undefined, {
        timeout: 10000,
      }),
    ).toBeDefined();
    // The API returns the frontmatter name, which the seed pins to the slug.
    const card = await screen.findByRole("link", { name: /Brand voice guide/ });
    expect(card.getAttribute("href")).toBe(
      "/marketplace/skills/brand-voice-guide",
    );
  });

  test("sits above the workflow catalogue, not below it", async () => {
    server.use(listing([brandVoice]));

    render(<MainMarkeplacePage />);

    const skills = await screen.findByText("AutoPilot Skills", undefined, {
      timeout: 10000,
    });
    const workflows = await screen.findByText("All AI Workflows");
    // Node.DOCUMENT_POSITION_FOLLOWING === 4: the workflows heading comes after.
    expect(skills.compareDocumentPosition(workflows) & 4).toBe(4);
  });

  test("shows the compatibility line only for a skill that needs one", async () => {
    server.use(listing([brandVoice, outreach]));

    render(<MainMarkeplacePage />);

    await screen.findAllByTestId("skill-card", undefined, { timeout: 10000 });
    const card = await screen.findByRole("link", { name: /Outreach playbook/ });
    expect(card.textContent).toContain("Works with");
    expect(card.textContent).toContain("Google");
    const brand = await screen.findByRole("link", {
      name: /Brand voice guide/,
    });
    expect(brand.textContent).not.toContain("Works with");
  });

  test("marks a skill the user already has as Added", async () => {
    server.use(
      listing([brandVoice, outreach]),
      // An install lands under the listing slug.
      getListCopilotSkillsMockHandler200([
        { name: "outreach-playbook", description: "Installed copy" },
      ]),
    );

    render(<MainMarkeplacePage />);

    const card = await screen.findByRole("link", { name: /Outreach playbook/ });
    await waitFor(() => expect(card.textContent).toContain("Added"));
    const brand = await screen.findByRole("link", {
      name: /Brand voice guide/,
    });
    expect(brand.textContent).not.toContain("Added");
    expect(brand.textContent).toContain("View");
  });

  test("offers Browse all only once the catalogue outgrows the shelf", async () => {
    server.use(listing([brandVoice, outreach], 2));

    const { unmount } = render(<MainMarkeplacePage />);

    await screen.findAllByTestId("skill-card", undefined, { timeout: 10000 });
    expect(
      screen.queryByRole("link", { name: /Browse all skills/ }),
    ).toBeNull();
    unmount();

    server.use(listing([brandVoice, outreach], 9));
    render(<MainMarkeplacePage />);

    expect(
      await screen.findByRole("link", { name: /Browse all skills/ }),
    ).toBeDefined();
  });

  test("keeps the header and offers a retry when the shelf fails", async () => {
    server.use(
      http.get("/api/proxy/api/store/skills", () =>
        HttpResponse.json({ detail: "Unavailable" }, { status: 500 }),
      ),
    );

    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("AutoPilot Skills", undefined, {
        timeout: 10000,
      }),
    ).toBeDefined();
    expect(await screen.findByRole("button", { name: "Retry" })).toBeDefined();
  });

  test("points a signed-in user at their own skills when none are published", async () => {
    server.use(listing([]));

    render(<MainMarkeplacePage />);

    expect(
      await screen.findByTestId("skills-shelf-empty", undefined, {
        timeout: 10000,
      }),
    ).toBeDefined();
  });

  test("says nothing at all to a visitor when there are no skills", async () => {
    mockUseAuth.mockReturnValue({ user: null, isLoggedIn: false });
    server.use(listing([]));

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("All AI Workflows")).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByText("AutoPilot Skills")).toBeNull(),
    );
  });

  test("stays hidden and fetches nothing outside the beta", async () => {
    flags.skillsHub = false;
    let requested = false;
    server.use(
      getGetV2ListMarketplaceSkillsMockHandler200(() => {
        requested = true;
        return {
          skills: [brandVoice],
          pagination: {
            total_items: 1,
            total_pages: 1,
            current_page: 1,
            page_size: 4,
          },
        };
      }),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("All AI Workflows")).toBeDefined();
    expect(screen.queryByText("AutoPilot Skills")).toBeNull();
    await waitFor(() => expect(requested).toBe(false));
  });

  test("does not paint the shelf before the flag has answered", async () => {
    flags.skillsHubReady = false;
    server.use(listing([brandVoice]));

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("All AI Workflows")).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByText("AutoPilot Skills")).toBeNull(),
    );
  });
});
