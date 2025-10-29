"use client";

import { use } from "react";
import { MainSearchResultPage } from "../components/MainSearchResultPage/MainSearchResultPage";

type MarketplaceSearchSort = "rating" | "runs" | "name" | "updated_at";
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
