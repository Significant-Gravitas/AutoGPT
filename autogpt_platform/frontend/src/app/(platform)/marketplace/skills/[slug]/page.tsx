import {
  getV2GetMarketplaceSkill,
  prefetchGetV2GetMarketplaceSkillQuery,
  prefetchGetV2ListMarketplaceSkillsQuery,
} from "@/app/api/__generated__/endpoints/store/store";
import { formatSkillTitle } from "../../components/SkillsSection/helpers";
import type { MarketplaceSkillDetails } from "@/app/api/__generated__/models/marketplaceSkillDetails";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { Metadata } from "next";
import { SkillPage } from "./components/SkillPage";

export const dynamic = "force-dynamic";

type PageParams = { slug: string };

export async function generateMetadata({
  params: _params,
}: {
  params: Promise<PageParams>;
}): Promise<Metadata> {
  const { slug } = await _params;
  // The generated client throws on any non-2xx, so an unknown or gated slug
  // would render a 500 here instead of the page's own not-found.
  try {
    const { data } = await getV2GetMarketplaceSkill(slug);
    const skill = data as MarketplaceSkillDetails;
    return {
      title: `${formatSkillTitle(skill.name)} - AutoGPT Marketplace`,
      description: skill.description,
    };
  } catch {
    return { title: "Skill - AutoGPT Marketplace" };
  }
}

export default async function MarketplaceSkillPage({
  params: _params,
}: {
  params: Promise<PageParams>;
}) {
  const { slug } = await _params;
  const queryClient = getQueryClient();
  await Promise.all([
    prefetchGetV2GetMarketplaceSkillQuery(queryClient, slug),
    // The "More skills" shelf under the body.
    prefetchGetV2ListMarketplaceSkillsQuery(queryClient, { page_size: 3 }),
  ]);

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <SkillPage slug={slug} />
    </HydrationBoundary>
  );
}
