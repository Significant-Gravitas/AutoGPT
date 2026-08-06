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
  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[], enabled: isLoggedIn },
  });

  const hiredTemplateIds = new Set(
    (expertsQuery.data ?? [])
      .map((expert) => expert.source_template_id)
      .filter((id): id is string => Boolean(id)),
  );

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
