import { useGetV2GetSpecificBlocks } from "@/app/api/__generated__/endpoints/default/default";
import type { GetV1GetExecutionDetails200 } from "@/app/api/__generated__/models/getV1GetExecutionDetails200";
import { okData } from "@/app/api/helpers";
import { formatStepName, getFailedSteps, getRunError } from "./helpers";

export function useRunFailures(run: GetV1GetExecutionDetails200) {
  const failedSteps = getFailedSteps(run);
  const blockIDs = Array.from(new Set(failedSteps.map((step) => step.blockID)));

  const blocksQuery = useGetV2GetSpecificBlocks(
    { block_ids: blockIDs },
    {
      query: {
        enabled: blockIDs.length > 0,
        select: okData,
        staleTime: Infinity,
      },
    },
  );

  const steps = failedSteps.map((step) => {
    const block = blocksQuery.data?.find(({ id }) => id === step.blockID);
    return { ...step, name: block ? formatStepName(block.name) : null };
  });

  return {
    steps,
    runError: steps.length > 0 ? null : getRunError(run),
    isLoadingNames: blocksQuery.isLoading,
  };
}
