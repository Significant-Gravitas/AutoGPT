import {
  getListExpertTemplatesMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetV1ListCredentialsMockHandler200 } from "@/app/api/__generated__/endpoints/integrations/integrations.msw";
import { getListCopilotSkillsMockHandler200 } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import {
  getGetV2GetMarketplaceSkillMockHandler200,
  getGetV2ListMarketplaceSkillsMockHandler200,
  getGetV2ListStoreAgentsMockHandler,
  getGetV2ListStoreAgentsResponseMock,
  getGetV2ListStoreCreatorsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import type { MarketplaceSkillDetails } from "@/app/api/__generated__/models/marketplaceSkillDetails";
import { server } from "@/mocks/mock-server";
import {
  configure,
  render,
  screen,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { MainMarkeplacePage } from "../components/MainMarketplacePage/MainMarketplacePage";

configure({ asyncUtilTimeout: 10000 });

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
      if (flag === "skills-hub" || flag === "hire-experts") return true;
      return actual.useGetFlag(flag as never);
    },
    useFlagStatus: (flag: string) => {
      if (flag === "skills-hub") return { enabled: true, ready: true };
      return actual.useFlagStatus(flag as never);
    },
  };
});

const outreach: MarketplaceSkill = {
  slug: "outreach-playbook",
  name: "outreach-playbook",
  title: "Outreach playbook",
  description: "Run cold outreach that gets replies.",
  categories: ["sales"],
  required_providers: [],
  install_count: 3,
  creator: null,
  creator_avatar: null,
};

const outreachDetails: MarketplaceSkillDetails = {
  ...outreach,
  skill_listing_version_id: "version-1",
  body: "# Outreach playbook\n\nFour sentences, no more.",
  triggers: [],
  updated_at: new Date("2026-09-07T00:00:00Z"),
};

const baseAgent = getGetV2ListStoreAgentsResponseMock().agents[0];
const agents = ["Lead Finder", "Inbox Sorter"].map((agent_name, i) => ({
  ...baseAgent,
  slug: `agent-${i}`,
  agent_name,
  creator: "AutoGPT",
}));

describe("Marketplace with hire-experts on", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getGetV2ListStoreAgentsMockHandler(
        getGetV2ListStoreAgentsResponseMock({ agents }),
      ),
      getGetV2ListStoreCreatorsMockHandler(),
      getListExpertTemplatesMockHandler([]),
      getListExpertsMockHandler([]),
      getListCopilotSkillsMockHandler200([]),
      getGetV1ListCredentialsMockHandler200([]),
      getGetV2ListMarketplaceSkillsMockHandler200({
        skills: [outreach],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 4,
        },
      }),
      getGetV2GetMarketplaceSkillMockHandler200(outreachDetails),
    );
  });

  test("lists skills as rows that do not link to a skill page", async () => {
    render(<MainMarkeplacePage />);

    const row = await screen.findByTestId("skill-row");
    expect(within(row).getByText("Outreach playbook")).toBeDefined();
    expect(within(row).queryByRole("link")).toBeNull();
    expect(screen.queryByTestId("skill-card")).toBeNull();
  });

  test("See skill opens the instructions in a dialog", async () => {
    render(<MainMarkeplacePage />);

    await userEvent.click(
      await screen.findByRole("button", {
        name: "More options for Outreach playbook",
      }),
    );
    await userEvent.click(
      await screen.findByRole("menuitem", { name: "See skill" }),
    );

    const dialog = await screen.findByTestId("skill-dialog");
    expect(
      await within(dialog).findByText("Four sentences, no more."),
    ).toBeDefined();
  });

  test("shows workflows as one row below skills, without the full grid", async () => {
    render(<MainMarkeplacePage />);

    const skills = await screen.findByRole("heading", { name: "Skills" });
    const workflows = await screen.findByRole("heading", {
      name: "Workflows",
    });
    // Node.DOCUMENT_POSITION_FOLLOWING === 4: the workflows heading comes after.
    expect(skills.compareDocumentPosition(workflows) & 4).toBe(4);
    expect(screen.getAllByTestId("workflow-chip")).toHaveLength(2);
    expect(screen.queryByText("All AI Workflows")).toBeNull();
    expect(screen.queryByTestId("store-card")).toBeNull();
  });

  test("Show all swaps the row for the full grid", async () => {
    render(<MainMarkeplacePage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Show all 2 workflows" }),
    );

    expect(screen.getAllByTestId("store-card").length).toBeGreaterThan(0);
    expect(screen.queryByTestId("workflow-chip")).toBeNull();
  });
});
