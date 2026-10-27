import {
  getGetBrainDumpSuggestedTasksQueryKey,
  useKickoffBrainDumpSuggestedTasks,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

// Asks the server, once per mount, to start the user's first suggested
// tasks as soon as `ready` turns true. The call is idempotent, so a page that
// asks again after the onboarding wizard already did never starts a task
// twice. Whatever it answers, the list is read again: a kickoff running in
// the background may have started the tasks while this one said "nothing".
export function useKickoffSuggestedTasks(ready: boolean) {
  const queryClient = useQueryClient();
  const hasAsked = useRef(false);
  const { mutate } = useKickoffBrainDumpSuggestedTasks({
    mutation: {
      onSettled: () => {
        void queryClient.invalidateQueries({
          queryKey: getGetBrainDumpSuggestedTasksQueryKey(),
        });
      },
    },
  });

  useEffect(() => {
    if (!ready || hasAsked.current) return;
    hasAsked.current = true;
    mutate();
  }, [ready, mutate]);
}
