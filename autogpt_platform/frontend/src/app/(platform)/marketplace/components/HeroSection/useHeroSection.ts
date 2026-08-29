import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useRouter } from "next/navigation";
import { DEFAULT_SEARCH_TERMS } from "./helpers";

export const useHeroSection = () => {
  const router = useRouter();
  const searchTerms = useGetFlag(Flag.MARKETPLACE_SEARCH_TERMS);

  function onFilterChange(selectedFilters: string[]) {
    const encodedTerm = encodeURIComponent(selectedFilters.join(", "));
    router.push(`/marketplace/search?searchTerm=${encodedTerm}`);
  }

  return {
    onFilterChange,
    searchTerms: searchTerms || DEFAULT_SEARCH_TERMS,
  };
};
