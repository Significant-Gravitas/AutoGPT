import {
  getListExpertLearnedNotesQueryKey,
  useArchiveExpertLearnedNote,
  useListExpertLearnedNotes,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Args {
  expertId: string | undefined;
}

export function useLearnedNotes({ expertId }: Args) {
  const isFeatureEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const queryClient = useQueryClient();
  const [deletingNoteId, setDeletingNoteId] = useState<string | null>(null);
  const enabled = isFeatureEnabled && Boolean(expertId);
  const { data, isLoading, isError } = useListExpertLearnedNotes(
    expertId ?? "",
    { query: { enabled, select: (response) => okData(response) ?? [] } },
  );
  const { mutate: archiveNote } = useArchiveExpertLearnedNote({
    mutation: {
      onSuccess: async () => {
        if (expertId) {
          await queryClient.invalidateQueries({
            queryKey: getListExpertLearnedNotesQueryKey(expertId),
          });
        }
        setDeletingNoteId(null);
      },
      onError: () => {
        setDeletingNoteId(null);
        toast({
          title: "Couldn't forget that note",
          description: "It's still here. Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  function forgetNote(noteId: string) {
    if (!expertId) return;
    setDeletingNoteId(noteId);
    archiveNote({ expertId, noteId });
  }

  return {
    isFeatureEnabled,
    notes: data ?? [],
    isLoading: enabled && isLoading,
    isError,
    deletingNoteId,
    forgetNote,
  };
}
