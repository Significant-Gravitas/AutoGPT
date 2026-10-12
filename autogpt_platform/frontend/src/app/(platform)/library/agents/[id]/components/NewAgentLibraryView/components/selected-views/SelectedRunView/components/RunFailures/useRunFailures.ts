import { useGetV2GetSpecificBlocks } from "@/app/api/__generated__/endpoints/default/default";
import type { GetV1GetExecutionDetails200 } from "@/app/api/__generated__/models/getV1GetExecutionDetails200";
import { okData } from "@/app/api/helpers";
import { getFailedStepCount, isCompletedWithErrors } from "@/lib/run-health";
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

  // The run's own tally still counts a failed step whose details are missing,
  // which keeps this notice in step with the run list and the status badge.
  const countedFailures = isCompletedWithErrors(run)
    ? getFailedStepCount(run)
    : 0;
  const failedStepCount = Math.max(steps.length, countedFailures);

  return {
    steps,
    failedStepCount,
    runError: failedStepCount > 0 ? null : getRunError(run),
    isLoadingNames: blocksQuery.isLoading,
  };
}
