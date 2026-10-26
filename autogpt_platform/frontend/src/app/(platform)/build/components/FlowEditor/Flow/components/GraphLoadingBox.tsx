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
      <div
        // TODO(tokens): no fitting role for border-gray-200 #E5E7EB; #E5E7EB is sidebar-border, whose .dark value (39 39 42) differs (QUALITY 43, SECRT-3154); dark-mode classes kept as written (QUALITY 43): no exact role: dark:border-gray-700 #374151, dark:bg-slate-800 #475263
        className="flex flex-col items-center gap-4 rounded-xlarge border border-gray-200 bg-surface p-8 shadow-lg dark:border-gray-700 dark:bg-slate-800"
      >
        <div className="relative h-12 w-12">
          {/* TODO(tokens): no fitting role for border-t-zinc-400 #ADADB3; #ADADB3 is foreground-subtle (a text colour); dark-mode classes kept as written (QUALITY 43): no exact role: dark:border-gray-700 #374151, dark:border-t-blue-400 #60A5FA */}
          <div className="absolute inset-0 animate-spin rounded-full border-4 border-separator border-t-zinc-400 dark:border-gray-700 dark:border-t-blue-400"></div>
        </div>
        <div className="flex flex-col items-center gap-2">
          {isSaving && <Text variant="h4">Saving Workflow</Text>}
          {flowContentLoading && <Text variant="h4">Loading Flow</Text>}

          {isSaving && (
            <Text variant="small">
              Please wait while we save your workflow...
            </Text>
          )}
          {flowContentLoading && (
            <Text variant="small">
              Please wait while we load your workflow...
            </Text>
          )}
        </div>
      </div>
    </div>
  );
};
