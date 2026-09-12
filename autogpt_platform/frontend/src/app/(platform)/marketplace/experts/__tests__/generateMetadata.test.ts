import { beforeEach, describe, expect, test, vi } from "vitest";

const mockListExpertTemplates = vi.hoisted(() => vi.fn());

vi.mock("@/app/api/__generated__/endpoints/experts/experts", () => ({
  listExpertTemplates: mockListExpertTemplates,
}));

vi.mock("../[expertId]/components/ExpertPage", () => ({
  ExpertPage: () => null,
}));

import { generateMetadata } from "../[expertId]/page";

const maria = {
  id: "template-maria",
  name: "Maria",
  role: "Marketing",
  tagline: "Turns your product story into campaigns that land.",
  bio: "A senior marketing strategist.",
  avatar_url: "/experts/maria.svg",
};

describe("generateMetadata", () => {
  beforeEach(() => {
    mockListExpertTemplates.mockReset();
  });

  test("previews the expert's name, role and tagline", async () => {
    mockListExpertTemplates.mockResolvedValue({ data: [maria] });

    const metadata = await generateMetadata({
      params: Promise.resolve({ expertId: "template-maria" }),
    });

    expect(metadata.title).toBe("Maria, Marketing - AutoGPT Marketplace");
    expect(metadata.openGraph).toMatchObject({
      title: "Maria, Marketing - AutoGPT Marketplace",
      description: "Turns your product story into campaigns that land.",
      type: "profile",
    });
    expect(metadata.alternates?.canonical).toContain(
      "/marketplace/experts/template-maria",
    );
  });

  test("never uses the SVG avatar, which no unfurler renders", async () => {
    mockListExpertTemplates.mockResolvedValue({ data: [maria] });

    const metadata = await generateMetadata({
      params: Promise.resolve({ expertId: "template-maria" }),
    });

    expect(metadata.openGraph).not.toHaveProperty("images");
    expect(metadata.twitter).toMatchObject({ card: "summary" });
  });

  test("falls back to a generic title for an unknown expert", async () => {
    mockListExpertTemplates.mockResolvedValue({ data: [maria] });

    const metadata = await generateMetadata({
      params: Promise.resolve({ expertId: "nobody" }),
    });

    expect(metadata.title).toBe("Expert - AutoGPT Marketplace");
  });

  test("falls back rather than throwing when the API is unreachable", async () => {
    mockListExpertTemplates.mockRejectedValue(new Error("fetch failed"));

    const metadata = await generateMetadata({
      params: Promise.resolve({ expertId: "template-maria" }),
    });

    expect(metadata.title).toBe("Expert - AutoGPT Marketplace");
  });
});
