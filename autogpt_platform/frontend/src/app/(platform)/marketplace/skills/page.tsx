import { prefetchGetV2ListMarketplaceSkillsInfiniteQuery } from "@/app/api/__generated__/endpoints/store/store";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { Metadata } from "next";
import { SkillsBrowsePage } from "./components/SkillsBrowsePage/SkillsBrowsePage";
import { BROWSE_PAGE_SIZE } from "../components/SkillsSection/helpers";

export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Skills - AutoGPT Marketplace",
  description:
    "Playbooks your experts follow. Add a playbook to your library, ready to assign to your experts.",
};

export default async function MarketplaceSkillsPage() {
  const queryClient = getQueryClient();

  // Only the unfiltered first page is worth prefetching: a category or search
  // arrives in the URL and would miss this key anyway.
  await prefetchGetV2ListMarketplaceSkillsInfiniteQuery(queryClient, {
    page: 1,
    page_size: BROWSE_PAGE_SIZE,
  });

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <SkillsBrowsePage />
    </HydrationBoundary>
  );
}
