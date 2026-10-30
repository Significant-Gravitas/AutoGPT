import {
  getGetRaiseIntroQueryOptions,
  prefetchGetRaiseIntroQuery,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import type { QueryClient } from "@tanstack/react-query";

// The opening is stateless and only depends on its inputs, so one answer
// serves the dialog's prefetch and the expert page for the whole session.
const INTRO_STALE_TIME = Infinity;

export function raiseIntroParams(name: string, jobTitle: string) {
  return {
    name: name.trim(),
    ...(jobTitle.trim() ? { job_title: jobTitle.trim() } : {}),
  };
}

/** Starts fetching the expert's opening while the dialog is still open, so
 *  the expert page already has it when it renders. */
export function prefetchRaiseIntro(
  queryClient: QueryClient,
  name: string,
  jobTitle: string,
) {
  if (!name.trim()) return;
  void prefetchGetRaiseIntroQuery(
    queryClient,
    raiseIntroParams(name, jobTitle),
    { query: { staleTime: INTRO_STALE_TIME } },
  );
}

export function raiseIntroQueryOptions(name: string, jobTitle: string) {
  return getGetRaiseIntroQueryOptions(raiseIntroParams(name, jobTitle), {
    query: {
      staleTime: INTRO_STALE_TIME,
      enabled: Boolean(name.trim()),
      select: (response) => okData(response) ?? null,
    },
  });
}
