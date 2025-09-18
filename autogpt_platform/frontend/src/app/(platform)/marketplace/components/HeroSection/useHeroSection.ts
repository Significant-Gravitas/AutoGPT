import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

export const useHeroSection = () => {
  const router = useRouter();
  const { completeStep } = useOnboarding();

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
  };
};
