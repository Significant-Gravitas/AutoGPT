"use client";

import { use } from "react";
import { SearchResults } from "../components/SearchResult/SearchResult";

type MarketplaceSearchPageSearchParams = { searchTerm?: string; sort?: string };

export default function MarketplaceSearchPage({
  searchParams,
}: {
  searchParams: Promise<MarketplaceSearchPageSearchParams>;
}) {
  return (
    // TODO : Fix sorting mechanism in here
    <SearchResults
      searchTerm={use(searchParams).searchTerm || ""}
      sort={use(searchParams).sort || "trending"}
    />
  );
}
