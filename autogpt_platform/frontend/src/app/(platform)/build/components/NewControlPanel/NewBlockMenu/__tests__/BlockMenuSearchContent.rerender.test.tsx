import { getGetV2BuilderSearchMockHandler } from "@/app/api/__generated__/endpoints/store/store.msw";
import type { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import type { SearchResponse } from "@/app/api/__generated__/models/searchResponse";
import { useBlockMenuStore } from "@/app/(platform)/build/stores/blockMenuStore";
import { server } from "@/mocks/mock-server";
import { act, render, waitFor } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { BlockMenuSearchContent } from "../BlockMenuSearchContent/BlockMenuSearchContent";
import { getSearchItemKeys } from "../BlockMenuSearchContent/helper";
import { highlightText } from "../helpers";

vi.mock("../helpers", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../helpers")>();
  return { ...actual, highlightText: vi.fn(actual.highlightText) };
});

vi.mock("../hooks/useAddBlockToBuilder", () => ({
  useAddBlockToBuilder: () => ({ addBlockWithPlacement: vi.fn() }),
}));

function findRow(id: string) {
  return waitFor(() => {
    const row = document.querySelector(`[data-id="block-card-${id}"]`);
    if (!row) throw new Error(`row ${id} not rendered`);
    return row;
  });
}

function makeBlock(id: string, name: string): BlockInfo {
  return {
    id,
    name,
    description: `${name} description`,
    inputSchema: {},
    outputSchema: {},
    costs: [],
    categories: [],
    contributors: [],
    staticOutput: false,
    uiType: "Standard",
  } as unknown as BlockInfo;
}

const blocks = [
  makeBlock("block-a", "Calculator Block"),
  makeBlock("block-b", "Calendar Block"),
  makeBlock("block-c", "Call Webhook Block"),
];

const searchResponse: SearchResponse = {
  items: blocks,
  search_id: "search-1",
  total_items: {
    blocks: 3,
    integrations: 0,
    marketplace_agents: 0,
    my_agents: 0,
  },
  pagination: {
    total_items: 3,
    total_pages: 1,
    current_page: 1,
    page_size: 8,
  },
} as SearchResponse;

beforeEach(() => {
  server.use(getGetV2BuilderSearchMockHandler(searchResponse));
  useBlockMenuStore.setState({
    searchQuery: "cal",
    searchId: undefined,
    filters: [],
    creators: [],
    creators_list: [],
  });
  vi.mocked(highlightText).mockClear();
});

describe("BlockMenuSearchContent", () => {
  test("filter counts and the creator list arriving do not re-render the result rows", async () => {
    render(<BlockMenuSearchContent />);

    await findRow("blocka");
    expect(await findRow("blockc")).toBeDefined();

    // Let the effects that copy the response into the store run first.
    await act(async () => {});
    const callsAfterLoad = vi.mocked(highlightText).mock.calls.length;
    expect(callsAfterLoad).toBeGreaterThan(0);

    act(() => {
      useBlockMenuStore.getState().setCategoryCounts({
        blocks: 9,
        integrations: 1,
        marketplace_agents: 2,
        my_agents: 3,
      });
      useBlockMenuStore.getState().setCreatorsList([]);
    });

    expect(vi.mocked(highlightText).mock.calls.length).toBe(callsAfterLoad);
    expect((await findRow("blockb")).textContent).toContain("Calendar");
  });

  test("a new search query still re-renders the rows with the new highlight", async () => {
    render(<BlockMenuSearchContent />);
    await findRow("blocka");
    await act(async () => {});
    vi.mocked(highlightText).mockClear();

    act(() => {
      useBlockMenuStore.getState().setSearchQuery("calc");
    });

    await waitFor(() =>
      expect(vi.mocked(highlightText)).toHaveBeenCalledWith(
        "Calculator",
        "calc",
      ),
    );
    expect((await findRow("blocka")).querySelector("mark")?.textContent).toBe(
      "Calc",
    );
  });
});

describe("getSearchItemKeys", () => {
  test("keys each result by type and id, and keeps repeats unique", () => {
    const keys = getSearchItemKeys([blocks[0], blocks[1], blocks[0]]);

    expect(keys).toEqual(["block-block-a", "block-block-b", "block-block-a-2"]);
    expect(new Set(keys).size).toBe(3);
  });

  test("a result keeps its key when an earlier one leaves the list", () => {
    const before = getSearchItemKeys(blocks);
    const after = getSearchItemKeys(blocks.slice(1));

    expect(after).toEqual(before.slice(1));
  });
});
