import type { DreamOperationsSnapshot } from "@/app/api/__generated__/models/dreamOperationsSnapshot";
import type { DreamPassResult } from "@/app/api/__generated__/models/dreamPassResult";
import type { DreamPassUsage } from "@/app/api/__generated__/models/dreamPassUsage";
import type { AnyJobStatus } from "./memoryJobStatus";

export interface DreamResult {
  passID: string;
  operations: DreamOperationsSnapshot;
  usage: DreamPassUsage;
}

// Only the sync_baseline path puts operations + usage on the result. The
// anthropic_batch callback writes a smaller summary (pass_id + stats), so
// there is no record to show for it.
export function getDreamResult(
  status: AnyJobStatus | undefined,
): DreamResult | undefined {
  if (status?.state !== "complete") return undefined;
  const result = status.result as DreamPassResult | null | undefined;
  if (!result?.operations || !result.usage) return undefined;
  return {
    passID: result.pass_id,
    operations: result.operations,
    usage: result.usage,
  };
}
