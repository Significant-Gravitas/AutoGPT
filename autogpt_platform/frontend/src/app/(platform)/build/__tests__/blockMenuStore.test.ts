import { describe, it, expect, beforeEach } from "vitest";
import { useBlockMenuStore } from "../stores/blockMenuStore";
import { DefaultStateType } from "../components/NewControlPanel/NewBlockMenu/types";
import { SearchEntryFilterAnyOfItem } from "@/app/api/__generated__/models/searchEntryFilterAnyOfItem";
import { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import { SearchResponseItemsItem } from "@/app/api/__generated__/models/searchResponseItemsItem";

beforeEach(() => {
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

describe("blockMenuStore", () => {
  describe("initial state", () => {
    it("has empty search and suggestion default state", () => {
      const state = useBlockMenuStore.getState();
      expect(state.searchQuery).toBe("");
      expect(state.searchId).toBeUndefined();
      expect(state.defaultState).toBe("suggestion");
      expect(state.integration).toBeUndefined();
      expect(state.filters).toEqual([]);
      expect(state.creators).toEqual([]);
    });
  });

  describe("search state", () => {
    it("sets search query", () => {
      useBlockMenuStore.getState().setSearchQuery("weather");
      expect(useBlockMenuStore.getState().searchQuery).toBe("weather");
    });

    it("sets search id", () => {
      useBlockMenuStore.getState().setSearchId("abc-123");
      expect(useBlockMenuStore.getState().searchId).toBe("abc-123");
    });

    it("clears search id", () => {
      useBlockMenuStore.getState().setSearchId("abc-123");
      useBlockMenuStore.getState().setSearchId(undefined);
      expect(useBlockMenuStore.getState().searchId).toBeUndefined();
    });
  });

  describe("default state", () => {
    it("sets default state", () => {
      useBlockMenuStore.getState().setDefaultState(DefaultStateType.ALL_BLOCKS);
      expect(useBlockMenuStore.getState().defaultState).toBe(
        DefaultStateType.ALL_BLOCKS,
      );
    });

    it("changes between states", () => {
      useBlockMenuStore
        .getState()
        .setDefaultState(DefaultStateType.INTEGRATIONS);
      useBlockMenuStore.getState().setDefaultState(DefaultStateType.MY_AGENTS);
      expect(useBlockMenuStore.getState().defaultState).toBe(
        DefaultStateType.MY_AGENTS,
      );
    });
  });

  describe("integration", () => {
    it("sets integration", () => {
      useBlockMenuStore.getState().setIntegration("slack");
      expect(useBlockMenuStore.getState().integration).toBe("slack");
    });

    it("clears integration", () => {
      useBlockMenuStore.getState().setIntegration("slack");
      useBlockMenuStore.getState().setIntegration(undefined);
      expect(useBlockMenuStore.getState().integration).toBeUndefined();
    });
  });

  describe("filters", () => {
    it("adds a filter", () => {
      useBlockMenuStore.getState().addFilter(SearchEntryFilterAnyOfItem.blocks);
      expect(useBlockMenuStore.getState().filters).toEqual(["blocks"]);
    });

    it("adds multiple filters", () => {
      useBlockMenuStore.getState().addFilter(SearchEntryFilterAnyOfItem.blocks);
      useBlockMenuStore
        .getState()
        .addFilter(SearchEntryFilterAnyOfItem.integrations);
      expect(useBlockMenuStore.getState().filters).toEqual([
        "blocks",
        "integrations",
      ]);
    });

    it("removes a filter", () => {
      useBlockMenuStore.getState().addFilter(SearchEntryFilterAnyOfItem.blocks);
      useBlockMenuStore
        .getState()
        .addFilter(SearchEntryFilterAnyOfItem.integrations);
      useBlockMenuStore
        .getState()
        .removeFilter(SearchEntryFilterAnyOfItem.blocks);
      expect(useBlockMenuStore.getState().filters).toEqual(["integrations"]);
    });

    it("sets filters directly", () => {
      useBlockMenuStore
        .getState()
        .setFilters([
          SearchEntryFilterAnyOfItem.my_agents,
          SearchEntryFilterAnyOfItem.marketplace_agents,
        ]);
      expect(useBlockMenuStore.getState().filters).toEqual([
        "my_agents",
        "marketplace_agents",
      ]);
    });

    it("removing a non-existent filter is a no-op", () => {
      useBlockMenuStore.getState().addFilter(SearchEntryFilterAnyOfItem.blocks);
      useBlockMenuStore
        .getState()
        .removeFilter(SearchEntryFilterAnyOfItem.integrations);
      expect(useBlockMenuStore.getState().filters).toEqual(["blocks"]);
    });
  });

  describe("creators", () => {
    it("adds a creator", () => {
      useBlockMenuStore.getState().addCreator("alice");
      expect(useBlockMenuStore.getState().creators).toEqual(["alice"]);
    });

    it("removes a creator", () => {
      useBlockMenuStore.getState().addCreator("alice");
      useBlockMenuStore.getState().addCreator("bob");
      useBlockMenuStore.getState().removeCreator("alice");
      expect(useBlockMenuStore.getState().creators).toEqual(["bob"]);
    });

    it("sets creators directly", () => {
      useBlockMenuStore.getState().setCreators(["x", "y"]);
      expect(useBlockMenuStore.getState().creators).toEqual(["x", "y"]);
    });
  });

  describe("setCreatorsList", () => {
    it("extracts creators from store_agent items", () => {
      const items: SearchResponseItemsItem[] = [
        {
          slug: "agent-1",
          agent_name: "Agent 1",
          creator: "alice",
        } as StoreAgent,
        {
          slug: "agent-2",
          agent_name: "Agent 2",
          creator: "bob",
        } as StoreAgent,
      ];

      useBlockMenuStore.getState().setCreatorsList(items);
      const list = useBlockMenuStore.getState().creators_list;
      expect(list).toContain("alice");
      expect(list).toContain("bob");
    });

    it("deduplicates creators across calls", () => {
      const items1 = [
        {
          slug: "a1",
          agent_name: "A1",
          creator: "alice",
        } as StoreAgent,
      ] as SearchResponseItemsItem[];
      const items2 = [
        {
          slug: "a2",
          agent_name: "A2",
          creator: "alice",
        } as StoreAgent,
      ] as SearchResponseItemsItem[];

      useBlockMenuStore.getState().setCreatorsList(items1);
      useBlockMenuStore.getState().setCreatorsList(items2);

      const aliceCount = useBlockMenuStore
        .getState()
        .creators_list.filter((c) => c === "alice").length;
      expect(aliceCount).toBe(1);
    });
  });

  describe("categoryCounts", () => {
    it("sets category counts", () => {
      const counts = {
        blocks: 10,
        integrations: 5,
        marketplace_agents: 3,
        my_agents: 2,
      };
      useBlockMenuStore.getState().setCategoryCounts(counts);
      expect(useBlockMenuStore.getState().categoryCounts).toEqual(counts);
    });
  });

  describe("reset", () => {
    it("resets search query, searchId, defaultState, and integration", () => {
      useBlockMenuStore.getState().setSearchQuery("test");
      useBlockMenuStore.getState().setSearchId("id-1");
      useBlockMenuStore.getState().setDefaultState(DefaultStateType.ALL_BLOCKS);
      useBlockMenuStore.getState().setIntegration("slack");
      useBlockMenuStore.getState().addFilter(SearchEntryFilterAnyOfItem.blocks);
      useBlockMenuStore.getState().addCreator("alice");

      useBlockMenuStore.getState().reset();

      const state = useBlockMenuStore.getState();
      expect(state.searchQuery).toBe("");
      expect(state.searchId).toBeUndefined();
      expect(state.defaultState).toBe("suggestion");
      expect(state.integration).toBeUndefined();
    });

    it("does not clear filters or creators", () => {
      useBlockMenuStore.getState().addFilter(SearchEntryFilterAnyOfItem.blocks);
      useBlockMenuStore.getState().addCreator("alice");

      useBlockMenuStore.getState().reset();

      expect(useBlockMenuStore.getState().filters).toEqual(["blocks"]);
      expect(useBlockMenuStore.getState().creators).toEqual(["alice"]);
    });
  });
});
