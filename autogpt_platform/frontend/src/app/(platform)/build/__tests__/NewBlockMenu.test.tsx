import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import { useBlockMenuStore } from "../stores/blockMenuStore";
import { useControlPanelStore } from "../stores/controlPanelStore";
import { DefaultStateType } from "../components/NewControlPanel/NewBlockMenu/types";
import { SearchEntryFilterAnyOfItem } from "@/app/api/__generated__/models/searchEntryFilterAnyOfItem";

// ---------------------------------------------------------------------------
// Mocks for heavy child components
// ---------------------------------------------------------------------------
vi.mock(
  "../components/NewControlPanel/NewBlockMenu/BlockMenuDefault/BlockMenuDefault",
  () => ({
    BlockMenuDefault: () => (
      <div data-testid="block-menu-default">Default Content</div>
    ),
  }),
);

vi.mock(
  "../components/NewControlPanel/NewBlockMenu/BlockMenuSearch/BlockMenuSearch",
  () => ({
    BlockMenuSearch: () => (
      <div data-testid="block-menu-search">Search Results</div>
    ),
  }),
);

// Mock query client used by the search bar hook
vi.mock("@/lib/react-query/queryClient", () => ({
  getQueryClient: () => ({
    invalidateQueries: vi.fn(),
  }),
}));

// ---------------------------------------------------------------------------
// Reset stores before each test
// ---------------------------------------------------------------------------
afterEach(() => {
  cleanup();
});

beforeEach(() => {
  useBlockMenuStore.getState().reset();
  useBlockMenuStore.setState({
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
  useControlPanelStore.getState().reset();
});

// ===========================================================================
// Section 1: blockMenuStore unit tests
// ===========================================================================
describe("blockMenuStore", () => {
  describe("searchQuery", () => {
    it("defaults to an empty string", () => {
      expect(useBlockMenuStore.getState().searchQuery).toBe("");
    });

    it("sets the search query", () => {
      useBlockMenuStore.getState().setSearchQuery("timer");
      expect(useBlockMenuStore.getState().searchQuery).toBe("timer");
    });
  });

  describe("defaultState", () => {
    it("defaults to SUGGESTION", () => {
      expect(useBlockMenuStore.getState().defaultState).toBe(
        DefaultStateType.SUGGESTION,
      );
    });

    it("sets the default state", () => {
      useBlockMenuStore.getState().setDefaultState(DefaultStateType.ALL_BLOCKS);
      expect(useBlockMenuStore.getState().defaultState).toBe(
        DefaultStateType.ALL_BLOCKS,
      );
    });
  });

  describe("filters", () => {
    it("defaults to an empty array", () => {
      expect(useBlockMenuStore.getState().filters).toEqual([]);
    });

    it("adds a filter", () => {
      useBlockMenuStore.getState().addFilter(SearchEntryFilterAnyOfItem.blocks);
      expect(useBlockMenuStore.getState().filters).toEqual([
        SearchEntryFilterAnyOfItem.blocks,
      ]);
    });

    it("removes a filter", () => {
      useBlockMenuStore
        .getState()
        .setFilters([
          SearchEntryFilterAnyOfItem.blocks,
          SearchEntryFilterAnyOfItem.integrations,
        ]);
      useBlockMenuStore
        .getState()
        .removeFilter(SearchEntryFilterAnyOfItem.blocks);
      expect(useBlockMenuStore.getState().filters).toEqual([
        SearchEntryFilterAnyOfItem.integrations,
      ]);
    });

    it("replaces all filters with setFilters", () => {
      useBlockMenuStore.getState().addFilter(SearchEntryFilterAnyOfItem.blocks);
      useBlockMenuStore
        .getState()
        .setFilters([SearchEntryFilterAnyOfItem.marketplace_agents]);
      expect(useBlockMenuStore.getState().filters).toEqual([
        SearchEntryFilterAnyOfItem.marketplace_agents,
      ]);
    });
  });

  describe("creators", () => {
    it("adds a creator", () => {
      useBlockMenuStore.getState().addCreator("alice");
      expect(useBlockMenuStore.getState().creators).toEqual(["alice"]);
    });

    it("removes a creator", () => {
      useBlockMenuStore.getState().setCreators(["alice", "bob"]);
      useBlockMenuStore.getState().removeCreator("alice");
      expect(useBlockMenuStore.getState().creators).toEqual(["bob"]);
    });

    it("replaces all creators with setCreators", () => {
      useBlockMenuStore.getState().addCreator("alice");
      useBlockMenuStore.getState().setCreators(["charlie"]);
      expect(useBlockMenuStore.getState().creators).toEqual(["charlie"]);
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

  describe("searchId", () => {
    it("defaults to undefined", () => {
      expect(useBlockMenuStore.getState().searchId).toBeUndefined();
    });

    it("sets and clears searchId", () => {
      useBlockMenuStore.getState().setSearchId("search-123");
      expect(useBlockMenuStore.getState().searchId).toBe("search-123");

      useBlockMenuStore.getState().setSearchId(undefined);
      expect(useBlockMenuStore.getState().searchId).toBeUndefined();
    });
  });

  describe("integration", () => {
    it("defaults to undefined", () => {
      expect(useBlockMenuStore.getState().integration).toBeUndefined();
    });

    it("sets the integration", () => {
      useBlockMenuStore.getState().setIntegration("slack");
      expect(useBlockMenuStore.getState().integration).toBe("slack");
    });
  });

  describe("reset", () => {
    it("resets searchQuery, searchId, defaultState, and integration", () => {
      useBlockMenuStore.getState().setSearchQuery("hello");
      useBlockMenuStore.getState().setSearchId("id-1");
      useBlockMenuStore.getState().setDefaultState(DefaultStateType.ALL_BLOCKS);
      useBlockMenuStore.getState().setIntegration("github");

      useBlockMenuStore.getState().reset();

      const state = useBlockMenuStore.getState();
      expect(state.searchQuery).toBe("");
      expect(state.searchId).toBeUndefined();
      expect(state.defaultState).toBe(DefaultStateType.SUGGESTION);
      expect(state.integration).toBeUndefined();
    });

    it("does not reset filters or creators (by design)", () => {
      useBlockMenuStore
        .getState()
        .setFilters([SearchEntryFilterAnyOfItem.blocks]);
      useBlockMenuStore.getState().setCreators(["alice"]);

      useBlockMenuStore.getState().reset();

      expect(useBlockMenuStore.getState().filters).toEqual([
        SearchEntryFilterAnyOfItem.blocks,
      ]);
      expect(useBlockMenuStore.getState().creators).toEqual(["alice"]);
    });
  });
});

// ===========================================================================
// Section 2: controlPanelStore unit tests
// ===========================================================================
describe("controlPanelStore", () => {
  it("defaults blockMenuOpen to false", () => {
    expect(useControlPanelStore.getState().blockMenuOpen).toBe(false);
  });

  it("sets blockMenuOpen", () => {
    useControlPanelStore.getState().setBlockMenuOpen(true);
    expect(useControlPanelStore.getState().blockMenuOpen).toBe(true);
  });

  it("sets forceOpenBlockMenu", () => {
    useControlPanelStore.getState().setForceOpenBlockMenu(true);
    expect(useControlPanelStore.getState().forceOpenBlockMenu).toBe(true);
  });

  it("resets all control panel state", () => {
    useControlPanelStore.getState().setBlockMenuOpen(true);
    useControlPanelStore.getState().setForceOpenBlockMenu(true);
    useControlPanelStore.getState().setSaveControlOpen(true);
    useControlPanelStore.getState().setForceOpenSave(true);

    useControlPanelStore.getState().reset();

    const state = useControlPanelStore.getState();
    expect(state.blockMenuOpen).toBe(false);
    expect(state.forceOpenBlockMenu).toBe(false);
    expect(state.saveControlOpen).toBe(false);
    expect(state.forceOpenSave).toBe(false);
  });
});

// ===========================================================================
// Section 3: BlockMenuContent integration tests
// ===========================================================================
// We import BlockMenuContent directly to avoid dealing with the Popover wrapper.
import { BlockMenuContent } from "../components/NewControlPanel/NewBlockMenu/BlockMenuContent/BlockMenuContent";

describe("BlockMenuContent", () => {
  it("shows BlockMenuDefault when there is no search query", () => {
    useBlockMenuStore.getState().setSearchQuery("");

    render(<BlockMenuContent />);

    expect(screen.getByTestId("block-menu-default")).toBeDefined();
    expect(screen.queryByTestId("block-menu-search")).toBeNull();
  });

  it("shows BlockMenuSearch when a search query is present", () => {
    useBlockMenuStore.getState().setSearchQuery("timer");

    render(<BlockMenuContent />);

    expect(screen.getByTestId("block-menu-search")).toBeDefined();
    expect(screen.queryByTestId("block-menu-default")).toBeNull();
  });

  it("renders the search bar", () => {
    render(<BlockMenuContent />);

    expect(
      screen.getByPlaceholderText(
        "Blocks, Agents, Integrations or Keywords...",
      ),
    ).toBeDefined();
  });

  it("switches from default to search view when store query changes", () => {
    const { rerender } = render(<BlockMenuContent />);
    expect(screen.getByTestId("block-menu-default")).toBeDefined();

    // Simulate typing by setting the store directly
    useBlockMenuStore.getState().setSearchQuery("webhook");
    rerender(<BlockMenuContent />);

    expect(screen.getByTestId("block-menu-search")).toBeDefined();
    expect(screen.queryByTestId("block-menu-default")).toBeNull();
  });

  it("switches back to default view when search query is cleared", () => {
    useBlockMenuStore.getState().setSearchQuery("something");
    const { rerender } = render(<BlockMenuContent />);
    expect(screen.getByTestId("block-menu-search")).toBeDefined();

    useBlockMenuStore.getState().setSearchQuery("");
    rerender(<BlockMenuContent />);

    expect(screen.getByTestId("block-menu-default")).toBeDefined();
    expect(screen.queryByTestId("block-menu-search")).toBeNull();
  });

  it("typing in the search bar updates the local input value", async () => {
    render(<BlockMenuContent />);

    const input = screen.getByPlaceholderText(
      "Blocks, Agents, Integrations or Keywords...",
    );
    fireEvent.change(input, { target: { value: "slack" } });

    expect((input as HTMLInputElement).value).toBe("slack");
  });

  it("shows clear button when input has text and clears on click", async () => {
    render(<BlockMenuContent />);

    const input = screen.getByPlaceholderText(
      "Blocks, Agents, Integrations or Keywords...",
    );
    fireEvent.change(input, { target: { value: "test" } });

    // The clear button should appear
    const clearButton = screen.getByRole("button");
    fireEvent.click(clearButton);

    await waitFor(() => {
      expect((input as HTMLInputElement).value).toBe("");
    });
  });
});
