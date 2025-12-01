import {
  getPostV1CreateNewGraphMutationOptions,
  getPutV1UpdateGraphVersionMutationOptions,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import { Text } from "@/components/atoms/Text/Text";
import { useIsMutating } from "@tanstack/react-query";

export const GraphLoadingBox = ({
  flowContentLoading,
}: {
  flowContentLoading: boolean;
}) => {
  const isCreating = useIsMutating({
    mutationKey: getPostV1CreateNewGraphMutationOptions().mutationKey,
  });
  const isUpdating = useIsMutating({
    mutationKey: getPutV1UpdateGraphVersionMutationOptions().mutationKey,
  });

  const isSaving = !!(isCreating || isUpdating);

  if (!flowContentLoading && !isSaving) {
    return null;
  }

  return (
    <div className="absolute left-[50%] top-[50%] z-[99] -translate-x-1/2 -translate-y-1/2">
      <div className="flex flex-col items-center gap-4 rounded-xlarge border border-gray-200 bg-white p-8 shadow-lg dark:border-gray-700 dark:bg-slate-800">
        <div className="relative h-12 w-12">
          <div className="absolute inset-0 animate-spin rounded-full border-4 border-zinc-100 border-t-zinc-400 dark:border-gray-700 dark:border-t-blue-400"></div>
        </div>
        <div className="flex flex-col items-center gap-2">
          {isSaving && <Text variant="h4">Saving Graph</Text>}
          {flowContentLoading && <Text variant="h4">Loading Flow</Text>}

          {isSaving && (
            <Text variant="small">Please wait while we save your graph...</Text>
          )}
          {flowContentLoading && (
            <Text variant="small">Please wait while we load your graph...</Text>
          )}
        </div>
      </div>
    </div>
  );
};
