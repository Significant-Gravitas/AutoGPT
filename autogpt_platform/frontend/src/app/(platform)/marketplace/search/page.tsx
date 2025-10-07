"use client";

import { use } from "react";
import { MainSearchResultPage } from "../components/MainSearchResultPage/MainSearchResultPage";

type MarketplaceSearchPageSearchParams = { searchTerm?: string; sort?: string };

export default function MarketplaceSearchPage({
  searchParams,
}: {
  searchParams: Promise<MarketplaceSearchPageSearchParams>;
}) {
  return (
    <MainSearchResultPage
      searchTerm={use(searchParams).searchTerm || ""}
      sort={use(searchParams).sort || "trending"}
    />
  );
}
