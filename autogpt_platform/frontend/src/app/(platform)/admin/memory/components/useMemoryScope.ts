"use client";

import { useEffect, useState } from "react";
import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useToast } from "@/components/molecules/Toast/use-toast";

export const AUTOPILOT_MEMORY_SCOPE = "autopilot";

export function useMemoryScope() {
  const { toast } = useToast();
  const [selectedScope, setSelectedScope] = useState(AUTOPILOT_MEMORY_SCOPE);
  const expertsQuery = useListExperts({
    query: {
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  const experts = expertsQuery.data;

  useEffect(() => {
    if (selectedScope === AUTOPILOT_MEMORY_SCOPE || expertsQuery.isLoading) {
      return;
    }
    if (
      expertsQuery.error ||
      !experts?.some(({ id }) => id === selectedScope)
    ) {
      setSelectedScope(AUTOPILOT_MEMORY_SCOPE);
      toast({
        title: "Expert no longer available",
        description: "Showing AutoPilot account memory instead.",
      });
    }
  }, [
    experts,
    expertsQuery.error,
    expertsQuery.isLoading,
    selectedScope,
    toast,
  ]);

  const selectedExpertID =
    selectedScope === AUTOPILOT_MEMORY_SCOPE ? undefined : selectedScope;

  return {
    selectedScope,
    setSelectedScope,
    selectedExpertID,
    experts: experts ?? [],
    expertsLoading: expertsQuery.isLoading,
    expertsError: expertsQuery.error,
  };
}
