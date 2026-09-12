import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockGetSpecificAgent = vi.hoisted(() => vi.fn());
const mockNotFound = vi.hoisted(() =>
  vi.fn(() => {
    throw new Error("NEXT_NOT_FOUND");
  }),
);
const mockGetServerUser = vi.hoisted(() => vi.fn());
const mockPrefetchLibrary = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({ notFound: mockNotFound }));

vi.mock("@/app/api/__generated__/endpoints/store/store", () => ({
  getV2GetSpecificAgent: mockGetSpecificAgent,
  prefetchGetV2GetSpecificAgentQuery: vi.fn(),
  prefetchGetV2ListStoreAgentsQuery: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/library/library", () => ({
  prefetchGetV2GetAgentByStoreIdQuery: mockPrefetchLibrary,
}));

vi.mock("@/lib/auth/server/getServerUser", () => ({
  getServerUser: mockGetServerUser,
}));

vi.mock("../../../../components/MainAgentPage/MainAgentPage", () => ({
  MainAgentPage: () => null,
}));

import MarketplaceAgentPage, { generateMetadata } from "../page";

const params = { creator: "pwuts", slug: "an-agent" };

describe("generateMetadata", () => {
  beforeEach(() => {
    mockGetSpecificAgent.mockReset();
    mockGetServerUser.mockResolvedValue({ user: null });
    mockNotFound.mockClear();
    mockPrefetchLibrary.mockClear();
  });

  describe.each([
    ["metadata", generateMetadata],
    ["page", MarketplaceAgentPage],
  ])("%s error handling", (_name, load) => {
    test("renders not-found for a missing listing", async () => {
      mockGetSpecificAgent.mockRejectedValue(
        new ApiError("Not Found", 404, null),
      );
      await expect(load({ params: Promise.resolve(params) })).rejects.toThrow(
        "NEXT_NOT_FOUND",
      );
      expect(mockNotFound).toHaveBeenCalledOnce();
    });

    test("preserves server errors", async () => {
      const error = new ApiError("Internal Server Error", 500, null);
      mockGetSpecificAgent.mockRejectedValue(error);
      await expect(load({ params: Promise.resolve(params) })).rejects.toBe(
        error,
      );
      expect(mockNotFound).not.toHaveBeenCalled();
    });

    test("preserves network errors", async () => {
      const error = new Error("fetch failed");
      mockGetSpecificAgent.mockRejectedValue(error);
      await expect(load({ params: Promise.resolve(params) })).rejects.toBe(
        error,
      );
      expect(mockNotFound).not.toHaveBeenCalled();
    });
  });

  test("prefetches the signed-in user's library entry for an existing listing", async () => {
    mockGetServerUser.mockResolvedValue({ user: { id: "user-1" } });
    mockGetSpecificAgent.mockResolvedValue({
      status: 200,
      data: { active_version_id: "version-1" },
    });
    await MarketplaceAgentPage({ params: Promise.resolve(params) });
    expect(mockPrefetchLibrary).toHaveBeenCalledWith(
      expect.anything(),
      "version-1",
      { query: { enabled: true } },
    );
  });

  test("previews the agent's own name, sub-heading and image", async () => {
    mockGetSpecificAgent.mockResolvedValue({
      data: {
        agent_name: "An Agent",
        sub_heading: "Summarises yesterday's runs",
        description: "- a bullet\n- another bullet\nand a long tail",
        agent_image: ["https://cdn.example.com/agent.png"],
      },
    });

    const metadata = await generateMetadata({
      params: Promise.resolve(params),
    });

    expect(metadata.title).toBe("An Agent - AutoGPT Marketplace");
    expect(metadata.openGraph?.title).toBe("An Agent - AutoGPT Marketplace");
    expect(metadata.openGraph?.description).toBe("Summarises yesterday's runs");
    expect(metadata.openGraph?.images).toEqual([
      "https://cdn.example.com/agent.png",
    ]);
    expect(metadata.twitter).toMatchObject({ card: "summary_large_image" });
  });

  test.each([
    ["missing", undefined],
    ["empty", ""],
  ])(
    "falls back to the agent name when sub_heading is %s",
    async (_label, sub_heading) => {
      mockGetSpecificAgent.mockResolvedValue({
        data: {
          agent_name: "An Agent",
          sub_heading,
          description: "The long listing description",
          agent_image: [],
        },
      });

      const metadata = await generateMetadata({
        params: Promise.resolve(params),
      });

      expect(metadata.openGraph?.description).toBe("An Agent");
    },
  );

  test("uses only the first image when the listing carries several", async () => {
    mockGetSpecificAgent.mockResolvedValue({
      data: {
        agent_name: "An Agent",
        description: "What the agent does",
        agent_image: ["https://cdn.example.com/1.png", "https://x/2.png"],
      },
    });

    const metadata = await generateMetadata({
      params: Promise.resolve(params),
    });

    expect(metadata.openGraph?.images).toEqual([
      "https://cdn.example.com/1.png",
    ]);
  });

  test("falls back to a text card when the listing has no image", async () => {
    mockGetSpecificAgent.mockResolvedValue({
      data: {
        agent_name: "An Agent",
        description: "What the agent does",
        agent_image: [],
      },
    });

    const metadata = await generateMetadata({
      params: Promise.resolve(params),
    });

    expect(metadata.openGraph).not.toHaveProperty("images");
    expect(metadata.twitter).toMatchObject({ card: "summary" });
  });
});
