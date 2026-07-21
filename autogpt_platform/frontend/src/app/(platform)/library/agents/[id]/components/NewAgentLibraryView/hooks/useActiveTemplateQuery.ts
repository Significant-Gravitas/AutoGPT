import { useGetV2GetASpecificPreset } from "@/app/api/__generated__/endpoints/presets/presets";
import { okData } from "@/app/api/helpers";
import { retryUnlessClientError } from "../helpers";

/**
 * The active template's preset detail, for prefilling the run modal.
 *
 * CAUTION — shared cache key: this uses the same generated endpoint (and
 * therefore the same React Query cache entry) as SelectedTriggerView's
 * preset detail fetch. All state exposed here is scoped to the Templates
 * tab so that a preset 404 on the Triggers tab (handled inline there)
 * can never leak out as a page-level template error.
 */
export function useActiveTemplateQuery(args: {
  activeItemId: string | null;
  activeTab: string;
}) {
  const onTemplatesTab = Boolean(
    args.activeTab === "templates" && args.activeItemId,
  );
  const query = useGetV2GetASpecificPreset(args.activeItemId ?? "", {
    query: {
      enabled: onTemplatesTab,
      select: okData,
      retry: retryUnlessClientError,
    },
  });

  return {
    activeTemplate:
      onTemplatesTab && query.isSuccess && query.data?.id === args.activeItemId
        ? query.data
        : null,
    isTemplateLoading: query.isLoading,
    templateError: onTemplatesTab ? (query.error ?? null) : null,
  };
}
