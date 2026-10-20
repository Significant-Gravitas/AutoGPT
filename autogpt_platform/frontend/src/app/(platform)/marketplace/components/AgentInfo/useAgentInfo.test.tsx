import { getPostV2AddMarketplaceAgentMockHandler } from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2DownloadAgentFileMockHandler } from "@/app/api/__generated__/endpoints/store/store.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import { act, renderHook } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useAgentInfo } from "./useAgentInfo";

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const sendDatafastEvent = vi.hoisted(() => vi.fn());
vi.mock("@/services/analytics", () => ({
  analytics: { sendDatafastEvent },
}));

vi.mock("@/lib/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/utils")>()),
  exportAsJSONFile: vi.fn(),
}));

afterEach(() => {
  sendDatafastEvent.mockClear();
  server.resetHandlers();
});

describe("useAgentInfo", () => {
  test("add_to_library carries the library ID and no workflow name", async () => {
    server.use(
      getPostV2AddMarketplaceAgentMockHandler({
        id: "lib-1",
        name: "Jane's private workflow",
      } as LibraryAgent),
    );
    const { result } = renderHook(
      () => useAgentInfo({ storeListingVersionId: "slv-1" }),
      { wrapper },
    );

    await act(() =>
      result.current.handleLibraryAction({ isAddingAgentFirstTime: true }),
    );

    expect(sendDatafastEvent).toHaveBeenCalledExactlyOnceWith(
      "add_to_library",
      { id: "lib-1" },
    );
  });

  test("download_agent carries the ID and no workflow name", async () => {
    server.use(getGetV2DownloadAgentFileMockHandler());
    const { result } = renderHook(
      () => useAgentInfo({ storeListingVersionId: "slv-1" }),
      { wrapper },
    );

    await act(() =>
      result.current.handleDownload("agent-1", "Jane's private workflow"),
    );

    expect(sendDatafastEvent).toHaveBeenCalledExactlyOnceWith(
      "download_agent",
      { id: "agent-1" },
    );
  });
});
