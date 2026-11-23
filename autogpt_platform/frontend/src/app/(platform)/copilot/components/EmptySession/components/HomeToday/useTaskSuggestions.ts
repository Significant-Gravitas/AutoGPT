import { useListTaskSuggestions } from "@/app/api/__generated__/endpoints/tasks/tasks";
import { okData } from "@/app/api/helpers";

/** The tasks Otto drafted that the user has neither started nor set aside. */
export function useTaskSuggestions() {
  const query = useListTaskSuggestions({
    query: { select: (res) => okData(res)?.tasks ?? [] },
  });
  return {
    suggestions: query.data ?? [],
    isLoaded: query.isSuccess,
  };
}
