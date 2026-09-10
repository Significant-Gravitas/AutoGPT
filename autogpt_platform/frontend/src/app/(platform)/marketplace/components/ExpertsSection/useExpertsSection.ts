import {
  useListExperts,
  useListExpertTemplates,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { useAuth } from "@/lib/auth/hooks/useAuth";

interface Args {
  category?: string | null;
  searchQuery?: string;
  /** False keeps the roster unfetched where the surface is hidden. */
  enabled?: boolean;
}

/** Templates are public, so the section can show them to anyone; only the
 *  hired roster (for the "Hired" state) needs a session. */
export function useExpertsSection({
  category,
  searchQuery,
  enabled = true,
}: Args = {}) {
  const { isLoggedIn } = useAuth();

  const templatesQuery = useListExpertTemplates(
    {
      ...(category ? { category } : {}),
      ...(searchQuery ? { search_query: searchQuery } : {}),
    },
    {
      query: {
        enabled,
        select: (x) => x.data as Expert[],
        // Keep the current cards on screen while a category change loads, so
        // picking a chip doesn't collapse the shelf to skeletons.
        placeholderData: (previousData) => previousData,
      },
    },
  );
  const expertsQuery = useListExperts({
    query: {
      select: (x) => x.data as Expert[],
      enabled: enabled && isLoggedIn,
    },
  });

  const hiredTemplateIds = new Set<string>();
  for (const expert of expertsQuery.data ?? []) {
    if (!expert.is_archived && expert.source_template_id) {
      hiredTemplateIds.add(expert.source_template_id);
    }
  }

  return {
    isLoggedIn,
    templates: templatesQuery.data ?? [],
    hiredTemplateIds,
    isLoading: templatesQuery.isLoading,
    isError: templatesQuery.isError,
  };
}
