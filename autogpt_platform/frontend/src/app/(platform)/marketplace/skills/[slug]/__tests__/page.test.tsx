import { getGetV2GetMarketplaceSkillMockHandler200 } from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MarketplaceSkillDetails } from "@/app/api/__generated__/models/marketplaceSkillDetails";
import { server } from "@/mocks/mock-server";
import { describe, expect, test, vi } from "vitest";
import MarketplaceSkillPage, { generateMetadata } from "../page";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/marketplace/skills/outreach-playbook",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({ slug: "outreach-playbook" }),
}));

const outreach = {
  slug: "outreach-playbook",
  name: "Outreach playbook",
  description: "Run cold outreach that gets replies.",
  categories: ["sales"],
  required_providers: ["google"],
  is_verified: true,
  install_count: 3,
  creator: null,
  creator_avatar: null,
  skill_listing_version_id: "version-1",
  body: "# Outreach playbook\n",
  triggers: ["cold email"],
  updated_at: new Date("2026-09-07T00:00:00Z"),
} as MarketplaceSkillDetails;

const params = Promise.resolve({ slug: "outreach-playbook" });

describe("marketplace skill page module", () => {
  test("titles the tab and describes the page from the listing itself", async () => {
    server.use(getGetV2GetMarketplaceSkillMockHandler200(outreach));

    const metadata = await generateMetadata({ params });

    expect(metadata.title).toBe("Outreach playbook - AutoGPT Marketplace");
    expect(metadata.description).toBe("Run cold outreach that gets replies.");
  });

  test("prefetches the listing so the page arrives populated", async () => {
    let requested = 0;
    server.use(
      getGetV2GetMarketplaceSkillMockHandler200(() => {
        requested += 1;
        return outreach;
      }),
    );

    const element = await MarketplaceSkillPage({ params });

    expect(requested).toBeGreaterThan(0);
    expect(element).toBeDefined();
  });
});
