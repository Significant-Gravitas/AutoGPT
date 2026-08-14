import {
  useListExperts,
  useListExpertTemplates,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useState } from "react";

export function useExpertsSection() {
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(
    null,
  );
  const { isLoggedIn } = useAuth();

  const templatesQuery = useListExpertTemplates({
    query: { select: (x) => x.data as Expert[], enabled: isLoggedIn },
  });
  const expertsQuery = useListExperts(undefined, {
    query: { select: (x) => x.data as Expert[], enabled: isLoggedIn },
  });

  const hiredTemplateIds = new Set<string>();
  for (const expert of expertsQuery.data ?? []) {
    if (!expert.is_archived && expert.source_template_id) {
      hiredTemplateIds.add(expert.source_template_id);
    }
  }

  return {
    templates: templatesQuery.data ?? [],
    hiredTemplateIds,
    isLoading: isLoggedIn && templatesQuery.isLoading,
    isError: templatesQuery.isError,
    selectedTemplateId,
    openTemplate: setSelectedTemplateId,
    closeSheet: () => setSelectedTemplateId(null),
  };
}
