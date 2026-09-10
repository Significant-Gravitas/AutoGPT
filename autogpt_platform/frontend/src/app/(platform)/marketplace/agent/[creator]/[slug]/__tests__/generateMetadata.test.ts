import { beforeEach, describe, expect, test, vi } from "vitest";

const mockGetSpecificAgent = vi.hoisted(() => vi.fn());

vi.mock("@/app/api/__generated__/endpoints/store/store", () => ({
  getV2GetSpecificAgent: mockGetSpecificAgent,
  prefetchGetV2GetSpecificAgentQuery: vi.fn(),
  prefetchGetV2ListStoreAgentsQuery: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/library/library", () => ({
  prefetchGetV2GetAgentByStoreIdQuery: vi.fn(),
}));

vi.mock("@/lib/auth/server/getServerUser", () => ({
  getServerUser: vi.fn(),
}));

vi.mock("../../../../components/MainAgentPage/MainAgentPage", () => ({
  MainAgentPage: () => null,
}));

import { generateMetadata } from "../page";

const params = { creator: "pwuts", slug: "an-agent" };

describe("generateMetadata", () => {
  beforeEach(() => {
    mockGetSpecificAgent.mockReset();
  });

  test("previews the agent's own name, description and image", async () => {
    mockGetSpecificAgent.mockResolvedValue({
      data: {
        agent_name: "An Agent",
        description: "What the agent does",
        agent_image: ["https://cdn.example.com/agent.png"],
      },
    });

    const metadata = await generateMetadata({
      params: Promise.resolve(params),
    });

    expect(metadata.title).toBe("An Agent - AutoGPT Marketplace");
    expect(metadata.openGraph?.title).toBe("An Agent - AutoGPT Marketplace");
    expect(metadata.openGraph?.description).toBe("What the agent does");
    expect(metadata.openGraph?.images).toEqual([
      "https://cdn.example.com/agent.png",
    ]);
    expect(metadata.twitter).toMatchObject({ card: "summary_large_image" });
  });

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
