import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { DEFAULT_SEARCH_TERMS } from "./helpers";

export const useHeroSection = () => {
  const router = useRouter();
  const { completeStep } = useOnboarding();
  const searchTerms = useGetFlag(Flag.MARKETPLACE_SEARCH_TERMS);

  // Mark marketplace visit task as completed
  useEffect(() => {
    completeStep("MARKETPLACE_VISIT");
  }, [completeStep]);

  function onFilterChange(selectedFilters: string[]) {
    const encodedTerm = encodeURIComponent(selectedFilters.join(", "));
    router.push(`/marketplace/search?searchTerm=${encodedTerm}`);
  }

  return {
    onFilterChange,
    searchTerms: searchTerms || DEFAULT_SEARCH_TERMS,
  };
};
