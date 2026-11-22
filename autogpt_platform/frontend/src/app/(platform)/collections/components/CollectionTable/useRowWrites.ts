import {
  useAcceptCollectionRowValues,
  useAddCollectionRow,
  useDeleteCollectionRow,
  useEditCollectionRow,
} from "@/app/api/__generated__/endpoints/collections/collections";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useQueryClient } from "@tanstack/react-query";
import type { TableRow } from "./helpers";

/** A person's writes to one collection's rows, each refreshing what shows it. */
export function useRowWrites(collectionID: string) {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  function refresh() {
    return queryClient.invalidateQueries({
      predicate: (query) => {
        const key = query.queryKey[0];
        return (
          typeof key === "string" &&
          (key.startsWith("/api/pages") || key.startsWith("/api/collections"))
        );
      },
    });
  }

  function onWriteError(error: unknown) {
    if (error instanceof ApiError && error.status === 409) {
      toast({
        title: "This row changed since you opened it",
        description: "It shows the new values now.",
      });
    } else {
      toast({
        title: "Couldn't save that",
        description: error instanceof Error ? error.message : undefined,
        variant: "destructive",
      });
    }
    refresh();
  }

  const mutation = { onSuccess: refresh, onError: onWriteError };
  const edit = useEditCollectionRow({ mutation });
  const add = useAddCollectionRow({ mutation });
  const remove = useDeleteCollectionRow({ mutation });
  const accept = useAcceptCollectionRowValues({ mutation });

  return {
    refresh,
    isSaving:
      edit.isPending || add.isPending || remove.isPending || accept.isPending,
    saveCell(row: TableRow, field: string, value: unknown) {
      return settled(
        edit.mutateAsync({
          collectionId: collectionID,
          rowId: row.id,
          data: { values: { [field]: value }, based_on: row.revision },
        }),
      );
    },
    addRow(values: Record<string, unknown>) {
      return settled(
        add.mutateAsync({ collectionId: collectionID, data: { values } }),
      );
    },
    deleteRow(row: TableRow) {
      return settled(
        remove.mutateAsync({ collectionId: collectionID, rowId: row.id }),
      );
    },
    acceptPending(row: TableRow, field: string) {
      return settled(
        accept.mutateAsync({
          collectionId: collectionID,
          rowId: row.id,
          data: { fields: [field], based_on: row.revision },
        }),
      );
    },
  };
}

/** Whether the write landed; a refusal has already been shown as a toast. */
async function settled(write: Promise<unknown>) {
  try {
    await write;
    return true;
  } catch {
    return false;
  }
}
