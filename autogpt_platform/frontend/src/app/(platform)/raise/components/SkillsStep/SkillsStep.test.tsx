import { getListCopilotSkillsMockHandler } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import {
  getGetV2ListMarketplaceSkillsMockHandler200,
  getGetV2ListStoreAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MarketplaceSkill } from "@/app/api/__generated__/models/marketplaceSkill";
import type { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { SkillsStep } from "./SkillsStep";

const flags = vi.hoisted(() => ({ skillsHub: true }));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) => {
      if (flag === "skills-hub")
        return { enabled: flags.skillsHub, ready: true };
      return actual.useFlagStatus(flag as never);
    },
  };
});

const storeAgent = {
  slug: "seo-writer",
  agent_name: "SEO Blog Writer",
  agent_image: "",
  creator: "acme",
  creator_avatar: "",
  sub_heading: "Writes optimized blog posts",
  description: "",
  runs: 10,
  rating: 5,
  agent_graph_id: "graph-1",
} as StoreAgent;

const outreach: MarketplaceSkill = {
  slug: "outreach-playbook",
  name: "outreach-playbook",
  title: "Outreach playbook",
  description: "Cold outreach that lands",
  categories: ["sales"],
  required_providers: [],
  install_count: 4,
};

function marketplaceSkills(skills: MarketplaceSkill[]) {
  return getGetV2ListMarketplaceSkillsMockHandler200({
    skills,
    pagination: {
      total_items: skills.length,
      total_pages: 1,
      current_page: 1,
      page_size: 3,
    },
  });
}

function renderSkills(
  overrides: Partial<Parameters<typeof SkillsStep>[0]> = {},
) {
  const onSubmit = vi.fn();
  const onSkip = vi.fn();
  render(
    <>
      <SkillsStep
        name="Otto"
        color="rose-300"
        submitted={null}
        existingCount={0}
        isSubmitting={false}
        onSubmit={onSubmit}
        onSkip={onSkip}
        {...overrides}
      />
      <Toaster />
    </>,
  );
  return { onSubmit, onSkip };
}

describe("SkillsStep", () => {
  beforeEach(() => {
    flags.skillsHub = true;
    server.use(marketplaceSkills([]));
  });

  test("shows only three default library skills", async () => {
    server.use(
      getListCopilotSkillsMockHandler([
        { name: "skill-a", description: "A" },
        { name: "skill-b", description: "B" },
        { name: "skill-c", description: "C" },
        { name: "skill-d", description: "D" },
      ]),
    );
    renderSkills();

    expect(await screen.findByText("skill-a")).toBeDefined();
    expect(screen.getByText("skill-b")).toBeDefined();
    expect(screen.getByText("skill-c")).toBeDefined();
    expect(screen.queryByText("skill-d")).toBeNull();
    expect(screen.getAllByRole("button", { name: "Add" })).toHaveLength(3);
  });

  test("adds a library skill without searching", async () => {
    server.use(
      getListCopilotSkillsMockHandler([
        { name: "seo-audit", description: "Audit landing pages" },
      ]),
    );
    const { onSubmit } = renderSkills();

    await userEvent.click(await screen.findByRole("button", { name: "Add" }));
    await userEvent.click(
      screen.getByRole("button", { name: "Bring Otto to life" }),
    );

    expect(onSubmit).toHaveBeenCalledWith([
      {
        kind: "skill",
        source: "library",
        id: "seo-audit",
        name: "seo-audit",
        marketplaceKey: undefined,
      },
    ]);
  });

  test("adds a marketplace skill by its slug", async () => {
    server.use(
      getListCopilotSkillsMockHandler([]),
      marketplaceSkills([outreach]),
    );
    const { onSubmit } = renderSkills();

    expect(await screen.findByText("Marketplace skill")).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Add" }));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Remove Outreach playbook/ }),
      ).toBeDefined(),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Bring Otto to life" }),
    );

    expect(onSubmit).toHaveBeenCalledWith([
      expect.objectContaining({
        kind: "skill",
        source: "marketplace",
        id: "outreach-playbook",
        name: "Outreach playbook",
      }),
    ]);
  });

  test("never offers a marketplace agent as a skill", async () => {
    server.use(
      getListCopilotSkillsMockHandler([]),
      getGetV2ListStoreAgentsMockHandler({
        agents: [storeAgent],
        pagination: {
          total_items: 1,
          total_pages: 1,
          current_page: 1,
          page_size: 3,
        },
      }),
      marketplaceSkills([outreach]),
    );
    renderSkills();

    expect(await screen.findByText("Outreach playbook")).toBeDefined();
    expect(screen.queryByText("SEO Blog Writer")).toBeNull();
  });

  test("offers no marketplace skills while the hub is off", async () => {
    flags.skillsHub = false;
    server.use(
      getListCopilotSkillsMockHandler([
        { name: "seo-audit", description: "Audit landing pages" },
      ]),
      marketplaceSkills([outreach]),
    );
    renderSkills();

    expect(await screen.findByText("seo-audit")).toBeDefined();
    expect(screen.queryByText("Outreach playbook")).toBeNull();
  });

  test("skip raises without extra skills", async () => {
    server.use(getListCopilotSkillsMockHandler([]));
    const { onSkip } = renderSkills();
    await userEvent.click(screen.getByRole("button", { name: "Skip" }));
    expect(onSkip).toHaveBeenCalled();
  });
});
