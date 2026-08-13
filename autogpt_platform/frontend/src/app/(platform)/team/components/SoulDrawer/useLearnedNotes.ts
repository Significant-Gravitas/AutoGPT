import {
  getGetExpertQueryKey,
  getListExpertsQueryKey,
  useDeleteExpertLearnedNote,
  useUpdateExpertLearnedNote,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

export function useLearnedNotes(expert: Expert | null) {
  const queryClient = useQueryClient();
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editText, setEditText] = useState("");

  const { mutateAsync: updateNote, isPending: isSaving } =
    useUpdateExpertLearnedNote();
  const { mutateAsync: deleteNote, isPending: isDeleting } =
    useDeleteExpertLearnedNote();

  // Newest first, mirroring the identity block the expert reads each session.
  const notes = [...(expert?.learned_notes ?? [])].reverse();

  function startEdit(noteId: string, fact: string) {
    setEditingId(noteId);
    setEditText(fact);
  }

  function cancelEdit() {
    setEditingId(null);
    setEditText("");
  }

  async function refresh() {
    if (!expert) return;
    await Promise.all([
      queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() }),
      queryClient.invalidateQueries({
        queryKey: getGetExpertQueryKey(expert.id),
      }),
    ]);
  }

  async function saveEdit() {
    const fact = editText.trim();
    if (!expert || !editingId || !fact) return;
    try {
      await updateNote({
        expertId: expert.id,
        noteId: editingId,
        data: { fact },
      });
      await refresh();
      toast({ title: "Note updated", variant: "success" });
      cancelEdit();
    } catch {
      toast({
        title: "Couldn't update note",
        description: "Your edit is still here. Please try again.",
        variant: "destructive",
      });
    }
  }

  async function removeNote(noteId: string) {
    if (!expert) return;
    try {
      await deleteNote({ expertId: expert.id, noteId });
      await refresh();
      toast({ title: "Note removed", variant: "success" });
    } catch {
      toast({ title: "Couldn't remove note", variant: "destructive" });
    }
  }

  return {
    notes,
    editingId,
    editText,
    setEditText,
    startEdit,
    cancelEdit,
    saveEdit,
    removeNote,
    isSaving,
    isDeleting,
  };
}
