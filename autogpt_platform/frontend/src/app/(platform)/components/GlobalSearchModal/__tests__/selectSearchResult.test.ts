import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { SearchResultItem } from "@/app/api/__generated__/models/searchResultItem";
import { selectSearchResult } from "../selectSearchResult";

function makeRouter() {
  const push = vi.fn();
  const router = { push } as unknown as AppRouterInstance;
  return { router, push };
}

function asItem(item: Partial<SearchResultItem>): SearchResultItem {
  return item as SearchResultItem;
}

describe("selectSearchResult", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("routes a chat_session to copilot via the sessionId query param", () => {
    const { router, push } = makeRouter();
    selectSearchResult(router, asItem({ id: "abc 1", type: "chat_session" }));
    expect(push).toHaveBeenCalledWith("/copilot?sessionId=abc%201");
  });

  it("routes a library_agent to its detail page", () => {
    const { router, push } = makeRouter();
    selectSearchResult(router, asItem({ id: "lib1", type: "library_agent" }));
    expect(push).toHaveBeenCalledWith("/library/agents/lib1");
  });

  it("routes a store_agent to the marketplace using creator and slug", () => {
    const { router, push } = makeRouter();
    selectSearchResult(
      router,
      asItem({
        id: "store1",
        type: "store_agent",
        metadata: { creator: "ac me", slug: "cool/agent" },
      }),
    );
    expect(push).toHaveBeenCalledWith(
      "/marketplace/agent/ac%20me/cool%2Fagent",
    );
  });

  it("does not route a store_agent missing creator or slug", () => {
    const { router, push } = makeRouter();
    selectSearchResult(
      router,
      asItem({ id: "store2", type: "store_agent", metadata: {} }),
    );
    expect(push).not.toHaveBeenCalled();
  });

  it("opens a workspace_file download in a new tab", () => {
    const { router, push } = makeRouter();
    const openSpy = vi.spyOn(window, "open").mockReturnValue(null);
    selectSearchResult(router, asItem({ id: "file1", type: "workspace_file" }));
    expect(openSpy).toHaveBeenCalledWith(
      "/api/proxy/api/workspace/files/file1/download",
      "_blank",
      "noopener,noreferrer",
    );
    expect(push).not.toHaveBeenCalled();
  });
});
