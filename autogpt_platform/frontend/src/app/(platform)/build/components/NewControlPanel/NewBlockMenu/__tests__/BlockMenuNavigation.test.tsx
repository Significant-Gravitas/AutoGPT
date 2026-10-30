import {
  getGetV2GetBuilderBlocksMockHandler,
  getGetV2GetBuilderIntegrationProvidersMockHandler,
  getGetV2GetBuilderItemCountsMockHandler,
  getGetV2GetBuilderSuggestionsMockHandler,
} from "@/app/api/__generated__/endpoints/default/default.msw";
import type { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { useBlockMenuStore } from "@/app/(platform)/build/stores/blockMenuStore";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { BlockMenuDefault } from "../BlockMenuDefault/BlockMenuDefault";
import { BlockMenuFilters } from "../BlockMenuFilters/BlockMenuFilters";
import { DefaultStateType } from "../types";

// Every block-menu view reads the store through its own selector; these tests
// drive the views through the store and the sidebar to check they still follow
// it.

vi.mock("../hooks/useAddBlockToBuilder", () => ({
  useAddBlockToBuilder: () => ({ addBlockWithPlacement: vi.fn() }),
}));

const pagination = {
  total_items: 1,
  total_pages: 1,
  current_page: 1,
  page_size: 10,
};

const githubBlock = {
  id: "github-block",
  name: "Github Read Issue Block",
  description: "Reads one GitHub issue",
  inputSchema: {},
  outputSchema: {},
  costs: [],
  categories: [],
  contributors: [],
  staticOutput: false,
  uiType: "Standard",
} as unknown as BlockInfo;

function findMenuItem(type: string) {
  return waitFor(() => {
    const item = document.querySelector(`[data-id="menu-item-${type}"]`);
    if (!item) throw new Error(`menu item ${type} not rendered`);
    return item;
  });
}

beforeEach(() => {
  server.use(
    getGetV2GetBuilderItemCountsMockHandler({
      all_blocks: 12,
      input_blocks: 2,
      action_blocks: 8,
      output_blocks: 2,
      integrations: 1,
      marketplace_agents: 0,
      my_agents: 0,
    }),
    getGetV2GetBuilderSuggestionsMockHandler({
      recent_searches: [],
      providers: [],
      top_blocks: [],
    }),
    getGetV2GetBuilderIntegrationProvidersMockHandler({
      providers: [
        { name: "github", description: "GitHub blocks", integration_count: 1 },
      ],
      pagination,
    }),
    getGetV2GetBuilderBlocksMockHandler({ blocks: [githubBlock], pagination }),
  );
  useBlockMenuStore.setState({
    searchQuery: "",
    searchId: undefined,
    defaultState: DefaultStateType.SUGGESTION,
    integration: undefined,
    filters: [],
    creators: [],
    creators_list: [],
    categoryCounts: {
      blocks: 0,
      integrations: 0,
      marketplace_agents: 0,
      my_agents: 0,
    },
  });
});

describe("block menu default view", () => {
  test("the sidebar opens integrations, one integration's blocks, and back", async () => {
    render(<BlockMenuDefault />);

    fireEvent.click(await findMenuItem("integrations"));
    expect(useBlockMenuStore.getState().defaultState).toBe(
      DefaultStateType.INTEGRATIONS,
    );

    fireEvent.click(await screen.findByText("GitHub blocks"));
    expect(useBlockMenuStore.getState().integration).toBe("github");
    expect(await screen.findByText("Reads one GitHub issue")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Integrations" }));
    expect(useBlockMenuStore.getState().integration).toBeUndefined();
    expect(await screen.findByText("GitHub blocks")).toBeDefined();
  });

  test("a store change made elsewhere switches the view", async () => {
    render(<BlockMenuDefault />);
    await findMenuItem("integrations");

    act(() => {
      useBlockMenuStore
        .getState()
        .setDefaultState(DefaultStateType.INTEGRATIONS);
      useBlockMenuStore.getState().setIntegration("github");
    });

    expect(await screen.findByText("Reads one GitHub issue")).toBeDefined();
  });
});

describe("block menu filters", () => {
  test("chips toggle filters and show the latest counts", async () => {
    render(<BlockMenuFilters />);

    act(() => {
      useBlockMenuStore.getState().setCategoryCounts({
        blocks: 7,
        integrations: 3,
        marketplace_agents: 0,
        my_agents: 0,
      });
    });
    fireEvent.mouseEnter(screen.getByRole("button", { name: /Blocks/ }));
    expect(await screen.findByText("7")).toBeDefined();

    fireEvent.click(screen.getByText("Blocks"));
    expect(useBlockMenuStore.getState().filters).toEqual(["blocks"]);

    fireEvent.click(screen.getByText("Blocks"));
    expect(useBlockMenuStore.getState().filters).toEqual([]);
  });
});
