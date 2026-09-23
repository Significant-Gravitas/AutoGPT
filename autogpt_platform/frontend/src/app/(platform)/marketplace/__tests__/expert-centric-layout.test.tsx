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
  getGetV2ListStoreCategoriesMockHandler,
  getGetV2ListStoreCreatorsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import type { MarketplaceSkillDetails } from "@/app/api/__generated__/models/marketplaceSkillDetails";
import { server } from "@/mocks/mock-server";
import { HttpResponse, http } from "msw";
import {
  configure,
  render,
  screen,
  waitFor,
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
// Ten so the shelf has more than the eight-tile preview to hold back.
const agents = Array.from({ length: 10 }, (_, i) => ({
  ...baseAgent,
  slug: `agent-${i}`,
  agent_graph_id: `graph-${i}`,
  agent_name: `Workflow ${i}`,
  creator: "AutoGPT",
}));

const agentsResponse = getGetV2ListStoreAgentsResponseMock({
  agents,
  pagination: {
    total_items: agents.length,
    total_pages: 1,
    current_page: 1,
    page_size: agents.length,
  },
});

describe("Marketplace with hire-experts on", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({ user: { id: "user-1" }, isLoggedIn: true });
    server.use(
      getGetV2ListStoreAgentsMockHandler(agentsResponse),
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

  test("lists skills as tiles that do not link to a skill page", async () => {
    render(<MainMarkeplacePage />);

    const tile = await screen.findByTestId("skill-tile");
    expect(within(tile).getByText("Outreach playbook")).toBeDefined();
    expect(tile.tagName).toBe("BUTTON");
    expect(screen.queryByTestId("skill-card")).toBeNull();
  });

  test("a skill tile opens the instructions in a dialog", async () => {
    render(<MainMarkeplacePage />);

    await userEvent.click(await screen.findByTestId("skill-tile"));

    const dialog = await screen.findByTestId("skill-dialog");
    expect(
      await within(dialog).findByText("Four sentences, no more."),
    ).toBeDefined();
  });

  test("shows eight workflow tiles below skills, without the full grid", async () => {
    render(<MainMarkeplacePage />);

    const skills = await screen.findByRole("heading", { name: "Skills" });
    const workflows = await screen.findByRole("heading", {
      name: "Workflows",
    });
    // Node.DOCUMENT_POSITION_FOLLOWING === 4: the workflows heading comes after.
    expect(skills.compareDocumentPosition(workflows) & 4).toBe(4);
    expect(screen.getAllByTestId("workflow-tile")).toHaveLength(8);
    expect(screen.queryByText("All AI Workflows")).toBeNull();
    expect(screen.queryByTestId("store-card")).toBeNull();
  });

  test("Load all asks the skills endpoint for the whole catalogue", async () => {
    const catalogue: MarketplaceSkill[] = Array.from(
      { length: 12 },
      (_, i) => ({
        ...outreach,
        slug: `skill-${i}`,
        name: `skill-${i}`,
        title: `Skill ${i}`,
      }),
    );
    server.use(
      http.get("/api/proxy/api/store/skills", ({ request }) => {
        const pageSize = Number(
          new URL(request.url).searchParams.get("page_size") ?? 8,
        );
        return HttpResponse.json({
          skills: catalogue.slice(0, pageSize),
          pagination: {
            total_items: catalogue.length,
            total_pages: 1,
            current_page: 1,
            page_size: pageSize,
          },
        });
      }),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findAllByTestId("skill-tile")).toHaveLength(8);
    await userEvent.click(
      await screen.findByRole("button", { name: "Load all 12 skills" }),
    );

    await waitFor(() =>
      expect(screen.getAllByTestId("skill-tile")).toHaveLength(12),
    );
  });

  test("a topic chip narrows the shelf to that category", async () => {
    const pipeline: MarketplaceSkill = {
      ...outreach,
      slug: "pipeline-review",
      name: "pipeline-review",
      title: "Pipeline review",
      categories: ["sales"],
    };
    server.use(
      getGetV2ListStoreCategoriesMockHandler([
        { value: "sales", label: "Sales", description: "Deals and outreach" },
        { value: "content", label: "Content", description: "Writing" },
      ]),
      http.get("/api/proxy/api/store/skills", ({ request }) => {
        const category = new URL(request.url).searchParams.get("category");
        const skills = category === "sales" ? [pipeline] : [outreach];
        return HttpResponse.json({
          skills,
          pagination: {
            total_items: skills.length,
            total_pages: 1,
            current_page: 1,
            page_size: 8,
          },
        });
      }),
    );

    render(<MainMarkeplacePage />);

    const chips = await screen.findByRole("group", {
      name: "Filter skills by topic",
    });
    await userEvent.click(within(chips).getByRole("button", { name: "Sales" }));

    expect(await screen.findByText("Pipeline review")).toBeDefined();
    await waitFor(() =>
      expect(screen.queryByText("Outreach playbook")).toBeNull(),
    );
  });

  test("Load all shows the rest as tiles, not cards", async () => {
    render(<MainMarkeplacePage />);

    await userEvent.click(
      await screen.findByRole("button", { name: "Load all 10 workflows" }),
    );

    expect(screen.getAllByTestId("workflow-tile")).toHaveLength(10);
    expect(screen.queryByTestId("store-card")).toBeNull();
    expect(screen.getByRole("button", { name: "Show fewer" })).toBeDefined();
  });

  test("keeps workflows with the same slug from different creators", async () => {
    const sharedSlugAgents = [
      {
        ...baseAgent,
        agent_graph_id: "graph-first",
        slug: "shared-slug",
        creator: "First creator",
        agent_name: "First workflow",
      },
      {
        ...baseAgent,
        agent_graph_id: "graph-second",
        slug: "shared-slug",
        creator: "Second creator",
        agent_name: "Second workflow",
      },
    ];
    server.use(
      getGetV2ListStoreAgentsMockHandler({
        agents: sharedSlugAgents,
        pagination: {
          total_items: sharedSlugAgents.length,
          total_pages: 1,
          current_page: 1,
          page_size: sharedSlugAgents.length,
        },
      }),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("First workflow")).toBeDefined();
    expect(screen.getByText("Second workflow")).toBeDefined();
  });

  test("does not promise all workflows when the response is capped", async () => {
    server.use(
      getGetV2ListStoreAgentsMockHandler({
        ...agentsResponse,
        pagination: { ...agentsResponse.pagination, total_items: 1_001 },
      }),
    );

    render(<MainMarkeplacePage />);

    expect(
      await screen.findByRole("button", { name: "Load 10 workflows" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Load all 10 workflows" }),
    ).toBeNull();
  });

  test("shows no runs when an older response omits the count", async () => {
    server.use(
      getGetV2ListStoreAgentsMockHandler({
        agents: [
          {
            ...baseAgent,
            agent_graph_id: "graph-without-runs",
            runs: null as unknown as number,
          },
        ],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 1,
        },
      }),
    );

    render(<MainMarkeplacePage />);

    expect(await screen.findByText("No runs")).toBeDefined();
  });
});
