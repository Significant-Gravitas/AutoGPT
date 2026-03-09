"use client";

import { useGetV1GetBusinessUnderstandingPrompts } from "@/app/api/__generated__/endpoints/auth/auth";
import { okData } from "@/app/api/helpers";
import { User } from "@supabase/supabase-js";
import { getQuickActions } from "./helpers";

export function useQuickActions(user?: User | null) {
  const quickPrompts = useGetV1GetBusinessUnderstandingPrompts({
    query: {
      enabled: Boolean(user),
      select: (response) => okData(response)?.prompts,
    },
  }).data;

  return getQuickActions(quickPrompts);
}
