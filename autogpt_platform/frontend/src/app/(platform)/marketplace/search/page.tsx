import { Suspense } from "react";
import { SearchBar } from "@/components/agptui/SearchBar";
import { SearchResults } from "./SearchResults";

// Enable ISR with 10-minute revalidation
export const revalidate = 600; // 10 minutes in seconds

// Generate static params for common search terms (optional optimization)
export async function generateStaticParams() {
  // You can pre-generate pages for common search terms
  return [
    { searchTerm: "seo" },
    { searchTerm: "marketing" },
    { searchTerm: "automation" },
    { searchTerm: "data" },
  ].map((params) => ({
    searchTerm: params.searchTerm,
  }));
}

type MarketplaceSearchPageSearchParams = {
  searchTerm?: string;
  sort?: string;
};

export default async function MarketplaceSearchPage({
  searchParams,
}: {
  searchParams: Promise<MarketplaceSearchPageSearchParams>;
}) {
  const params = await searchParams;
  const searchTerm = params.searchTerm || "";
  const sort = params.sort || "trending";

  return (
    <div className="w-full">
      <div className="mx-auto min-h-screen max-w-[1440px] px-10 lg:min-w-[1440px]">
        <div className="mt-8 flex items-center">
          <div className="flex-1">
            <h2 className="text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
              Results for:
            </h2>
            <h1 className="font-poppins text-2xl font-semibold leading-[32px] text-neutral-800 dark:text-neutral-100">
              {searchTerm}
            </h1>
          </div>
          <div className="flex-none">
            <SearchBar width="w-[439px]" height="h-[60px]" />
          </div>
        </div>

        <Suspense
          fallback={
            <div className="mt-20 flex flex-col items-center justify-center">
              <p className="text-neutral-500 dark:text-neutral-400">
                Loading...
              </p>
            </div>
          }
        >
          <SearchResults searchTerm={searchTerm} sort={sort} />
        </Suspense>
      </div>
    </div>
  );
}
