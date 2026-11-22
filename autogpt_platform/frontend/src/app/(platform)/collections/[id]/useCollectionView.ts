import {
  getListCollectionsQueryKey,
  useDeleteCollection,
  useGetCollection,
  useListCollectionRows,
} from "@/app/api/__generated__/endpoints/collections/collections";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useState } from "react";

export const COLLECTIONS_LIST_HREF = "/pages?tab=collections";

export function useCollectionView(collectionID: string) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const collection = useGetCollection(collectionID, {
    query: { retry: false, select: okData },
  });
  const rows = useListCollectionRows(
    collectionID,
    { limit: 50 },
    { query: { retry: false, select: okData } },
  );
  const remove = useDeleteCollection({
    mutation: {
      onSuccess: async () => {
        await queryClient.invalidateQueries({
          queryKey: getListCollectionsQueryKey(),
        });
        router.push(COLLECTIONS_LIST_HREF);
      },
      onError: () =>
        toast({
          title: "Couldn't delete the collection",
          variant: "destructive",
        }),
    },
  });

  return {
    collection: collection.data?.collection,
    limits: collection.data?.limits,
    rows: rows.data?.rows ?? [],
    total: rows.data?.total ?? 0,
    isLoading: collection.isLoading || rows.isLoading,
    isMissing: collection.isError,
    confirmingDelete,
    setConfirmingDelete,
    isDeleting: remove.isPending,
    deleteCollection: () => remove.mutate({ collectionId: collectionID }),
  };
}
