import { useListExpertTemplates } from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";

interface Args {
  expertId: string;
}

/** The template behind a marketplace expert page. Templates are public, so
 *  every visitor sees the same profile; hiring is not open yet. */
export function useExpertPage({ expertId }: Args) {
  const templatesQuery = useListExpertTemplates({
    query: { select: (x) => x.data as Expert[] },
  });

  const expert =
    (templatesQuery.data ?? []).find((template) => template.id === expertId) ??
    null;

  return {
    expert,
    isLoading: templatesQuery.isLoading,
    isError: templatesQuery.isError,
    refetch: templatesQuery.refetch,
  };
}
