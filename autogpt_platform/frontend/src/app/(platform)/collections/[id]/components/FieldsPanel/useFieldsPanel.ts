import { useUpdateCollection } from "@/app/api/__generated__/endpoints/collections/collections";
import type { CollectionUpdateRequest } from "@/app/api/__generated__/models/collectionUpdateRequest";
import type { FieldType } from "@/app/api/__generated__/models/fieldType";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";

export function useFieldsPanel(collectionID: string) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const update = useUpdateCollection({
    mutation: {
      onSuccess: () =>
        queryClient.invalidateQueries({
          predicate: (query) =>
            typeof query.queryKey[0] === "string" &&
            query.queryKey[0].startsWith("/api/collections"),
        }),
      onError: (error) =>
        toast({
          title: "Couldn't change the field",
          description: error instanceof Error ? error.message : undefined,
          variant: "destructive",
        }),
    },
  });

  function change(data: CollectionUpdateRequest) {
    update.mutate({ collectionId: collectionID, data });
  }

  return {
    isSaving: update.isPending,
    rename(field: string, to: string) {
      const name = to.trim();
      if (name && name !== field)
        change({ rename_fields: [{ field, to: name }] });
    },
    retype(field: string, type: FieldType) {
      change({ change_types: [{ field, type }] });
    },
  };
}
