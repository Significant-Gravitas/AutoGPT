"use client";

import {
  getGetV2AdminListModelsQueryKey,
  getGetV2AdminListProvidersQueryKey,
  getGetV2ListCreatorsQueryKey,
  getGetV2ListMigrationsQueryKey,
  getGetV2ListRoutesQueryKey,
  useDeleteV2DeleteCreator,
  useDeleteV2DeleteModel,
  usePatchV2UpdateCreator,
  usePatchV2UpdateModel,
  usePostV2CreateCreator,
  usePostV2CreateModel,
  usePostV2RevertMigration,
  usePostV2ToggleModel,
  usePutV2SetRoute,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";

function errorDescription(error: unknown): string {
  if (error && typeof error === "object" && "message" in error) {
    return String((error as { message: unknown }).message);
  }
  return "Request failed — see console for details";
}

export function useLlmRegistryMutations() {
  const queryClient = useQueryClient();
  const { toast } = useToast();

  function invalidateModels() {
    queryClient.invalidateQueries({
      queryKey: getGetV2AdminListModelsQueryKey({ page: 1, page_size: 200 }),
    });
    queryClient.invalidateQueries({
      queryKey: getGetV2AdminListProvidersQueryKey(),
    });
  }

  function onError(title: string) {
    return (error: unknown) => {
      toast({
        title,
        description: errorDescription(error),
        variant: "destructive",
      });
    };
  }

  const createModel = usePostV2CreateModel({
    mutation: {
      onSuccess: invalidateModels,
      onError: onError("Failed to create model"),
    },
  });
  const updateModel = usePatchV2UpdateModel({
    mutation: {
      onSuccess: invalidateModels,
      onError: onError("Failed to update model"),
    },
  });
  const toggleModel = usePostV2ToggleModel({
    mutation: {
      onSuccess: () => {
        invalidateModels();
        queryClient.invalidateQueries({
          queryKey: getGetV2ListMigrationsQueryKey({ include_reverted: true }),
        });
      },
      onError: onError("Failed to toggle model"),
    },
  });
  const deleteModel = useDeleteV2DeleteModel({
    mutation: {
      onSuccess: () => {
        invalidateModels();
        queryClient.invalidateQueries({
          queryKey: getGetV2ListMigrationsQueryKey({ include_reverted: true }),
        });
      },
      onError: onError("Failed to delete model"),
    },
  });

  const createCreator = usePostV2CreateCreator({
    mutation: {
      onSuccess: () =>
        queryClient.invalidateQueries({
          queryKey: getGetV2ListCreatorsQueryKey(),
        }),
      onError: onError("Failed to create creator"),
    },
  });
  const updateCreator = usePatchV2UpdateCreator({
    mutation: {
      onSuccess: () =>
        queryClient.invalidateQueries({
          queryKey: getGetV2ListCreatorsQueryKey(),
        }),
      onError: onError("Failed to update creator"),
    },
  });
  const deleteCreator = useDeleteV2DeleteCreator({
    mutation: {
      onSuccess: () =>
        queryClient.invalidateQueries({
          queryKey: getGetV2ListCreatorsQueryKey(),
        }),
      onError: onError("Failed to delete creator"),
    },
  });

  const setRoute = usePutV2SetRoute({
    mutation: {
      onSuccess: (response) => {
        queryClient.invalidateQueries({
          queryKey: getGetV2ListRoutesQueryKey(),
        });
        const warnings =
          response.status === 200 ? (response.data.warnings ?? []) : [];
        for (const warning of warnings) {
          toast({ title: "Routing warning", description: warning });
        }
      },
      onError: onError("Failed to update routing cell"),
    },
  });

  const revertMigration = usePostV2RevertMigration({
    mutation: {
      onSuccess: () => {
        invalidateModels();
        queryClient.invalidateQueries({
          queryKey: getGetV2ListMigrationsQueryKey({ include_reverted: true }),
        });
      },
      onError: onError("Failed to revert migration"),
    },
  });

  return {
    createModel,
    updateModel,
    toggleModel,
    deleteModel,
    createCreator,
    updateCreator,
    deleteCreator,
    setRoute,
    revertMigration,
  };
}
