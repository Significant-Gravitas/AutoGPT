import {
  useListExperts,
  useListExpertTemplates,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useState } from "react";
import { getHiredExpertsLookup } from "./helpers";

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

  const hiredLookup = getHiredExpertsLookup(expertsQuery.data, expertsQuery);

  return {
    templates: templatesQuery.data ?? [],
    hiredTemplateIds: new Set(hiredLookup.byTemplateId.keys()),
    isHiredLookupUnresolved: isLoggedIn && hiredLookup.state === "loading",
    isHiredLookupError: isLoggedIn && hiredLookup.state === "error",
    isLoading: isLoggedIn && templatesQuery.isLoading,
    isError: templatesQuery.isError,
    selectedTemplateId,
    openTemplate: setSelectedTemplateId,
    closeSheet: () => setSelectedTemplateId(null),
  };
}
