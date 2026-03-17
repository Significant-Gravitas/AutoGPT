"use client";

import { use } from "react";
import { MainSearchResultPage } from "../components/MainSearchResultPage/MainSearchResultPage";
import { GetV2ListStoreAgentsParams } from "@/app/api/__generated__/models/getV2ListStoreAgentsParams";

type MarketplaceSearchSort = GetV2ListStoreAgentsParams["sorted_by"];
type MarketplaceSearchPageSearchParams = {
  searchTerm?: string;
  sort?: MarketplaceSearchSort;
};

export default function MarketplaceSearchPage({
  searchParams,
}: {
  searchParams: Promise<MarketplaceSearchPageSearchParams>;
}) {
  return (
    <MainSearchResultPage
      searchTerm={use(searchParams).searchTerm || ""}
      sort={use(searchParams).sort || "runs"}
    />
  );
}
