import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockGetCreatorDetails = vi.hoisted(() => vi.fn());
const mockNotFound = vi.hoisted(() =>
  vi.fn(() => {
    throw new Error("NEXT_NOT_FOUND");
  }),
);

vi.mock("@/app/api/__generated__/endpoints/store/store", () => ({
  getV2GetCreatorDetails: mockGetCreatorDetails,
  prefetchGetV2GetCreatorDetailsQuery: vi.fn(),
  prefetchGetV2ListStoreAgentsQuery: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  notFound: mockNotFound,
}));

vi.mock("../components/MainCreatorPage/MainCreatorPage", () => ({
  MainCreatorPage: () => null,
}));

import { generateMetadata } from "../page";

describe("generateMetadata", () => {
  beforeEach(() => {
    mockGetCreatorDetails.mockReset();
    mockNotFound.mockClear();
  });

  test("returns creator metadata on success", async () => {
    mockGetCreatorDetails.mockResolvedValue({
      data: { name: "Creator One", description: "Creator profile" },
    });

    const metadata = await generateMetadata({
      params: Promise.resolve({ creator: "Creator-One" }),
    });

    expect(mockGetCreatorDetails).toHaveBeenCalledWith("creator-one");
    expect(metadata.title).toBe("Creator One - AutoGPT Store");
    expect(metadata.description).toBe("Creator profile");
  });

  test("renders the 404 page when the creator does not exist", async () => {
    mockGetCreatorDetails.mockRejectedValue(
      new ApiError("Not Found", 404, undefined),
    );

    await expect(
      generateMetadata({ params: Promise.resolve({ creator: "missing" }) }),
    ).rejects.toThrow("NEXT_NOT_FOUND");
    expect(mockNotFound).toHaveBeenCalled();
  });

  test("rethrows non-404 API errors", async () => {
    const serverError = new ApiError("Internal Server Error", 500, undefined);
    mockGetCreatorDetails.mockRejectedValue(serverError);

    await expect(
      generateMetadata({ params: Promise.resolve({ creator: "someone" }) }),
    ).rejects.toBe(serverError);
    expect(mockNotFound).not.toHaveBeenCalled();
  });

  test("rethrows non-API errors", async () => {
    const networkError = new Error("fetch failed");
    mockGetCreatorDetails.mockRejectedValue(networkError);

    await expect(
      generateMetadata({ params: Promise.resolve({ creator: "someone" }) }),
    ).rejects.toBe(networkError);
    expect(mockNotFound).not.toHaveBeenCalled();
  });
});
