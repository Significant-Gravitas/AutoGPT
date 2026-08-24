import { getListCopilotSkillsMockHandler } from "@/app/api/__generated__/endpoints/skills/skills.msw";
import {
  getGetV2GetSpecificAgentMockHandler,
  getGetV2ListStoreAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import type { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, test, vi } from "vitest";
import { SkillsStep } from "./SkillsStep";

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

  test("searches marketplace agents as skills", async () => {
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
      getGetV2GetSpecificAgentMockHandler({
        store_listing_version_id: "listing-version-42",
        slug: "seo-writer",
        agent_name: "SEO Blog Writer",
        creator: "acme",
      } as StoreAgentDetails),
    );
    const { onSubmit } = renderSkills();

    await userEvent.type(
      screen.getByRole("textbox", { name: "Search skills" }),
      "seo",
    );

    expect(await screen.findByText("Marketplace skill")).toBeDefined();
    await userEvent.click(screen.getByRole("button", { name: "Add" }));
    await waitFor(() =>
      expect(
        screen.getByRole("button", { name: /Remove SEO Blog Writer/ }),
      ).toBeDefined(),
    );
    await userEvent.click(
      screen.getByRole("button", { name: "Bring Otto to life" }),
    );

    expect(onSubmit).toHaveBeenCalledWith([
      expect.objectContaining({
        kind: "skill",
        source: "marketplace",
        id: "listing-version-42",
        name: "SEO Blog Writer",
      }),
    ]);
  });

  test("skip raises without extra skills", async () => {
    server.use(getListCopilotSkillsMockHandler([]));
    const { onSkip } = renderSkills();
    await userEvent.click(screen.getByRole("button", { name: "Skip" }));
    expect(onSkip).toHaveBeenCalled();
  });
});
