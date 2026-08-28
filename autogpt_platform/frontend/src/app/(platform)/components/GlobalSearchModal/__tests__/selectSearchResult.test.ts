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
    expect(push).toHaveBeenCalledWith(
      "/copilot?sessionId=abc+1&organizationId=__personal__&teamId=__org_home__",
    );
  });

  it("routes a library_agent to its detail page", () => {
    const { router, push } = makeRouter();
    selectSearchResult(router, asItem({ id: "lib1", type: "library_agent" }));
    expect(push).toHaveBeenCalledWith(
      "/library/agents/lib1?organizationId=__personal__&teamId=__org_home__",
    );
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

  it("opens a tenant-scoped workspace_file download in a new tab", async () => {
    const { router, push } = makeRouter();
    const replace = vi.fn();
    const target = {
      close: vi.fn(),
      location: { replace },
    } as unknown as Window;
    const openSpy = vi.spyOn(window, "open").mockReturnValue(target);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("workspace file", {
        status: 200,
        headers: { "content-type": "text/plain" },
      }),
    );
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:workspace-file");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    selectSearchResult(router, asItem({ id: "file1", type: "workspace_file" }));
    expect(openSpy).toHaveBeenCalledWith(
      "about:blank",
      "_blank",
      "noopener,noreferrer",
    );
    await vi.waitFor(() => {
      expect(fetchSpy).toHaveBeenCalledWith(
        "/api/proxy/api/workspace/files/file1/download",
        {
          headers: { "X-Org-Id": "", "X-Team-Id": "" },
        },
      );
      expect(replace).toHaveBeenCalledWith("blob:workspace-file");
    });
    expect(push).not.toHaveBeenCalled();
  });
});
