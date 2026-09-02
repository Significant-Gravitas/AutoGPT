import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useMemo } from "react";
import { parseGraphID, parseGraphExecutionID } from "@/lib/graph-ids";
import type { GraphID, GraphExecutionID } from "@/lib/autogpt-server-api/types";

/**
 * Canonical Builder URL state.
 *
 * Single owner for flowID / flowVersion / flowExecutionID.
 * All Builder surfaces MUST import from here — do not declare
 * `useQueryStates({flowID...})` elsewhere (REL-004).
 *
 * Raw nuqs strings are validated as UUIDs before becoming trusted
 * branded IDs. Malformed URL values become null and no subscription /
 * graph fetch fires (REL-003).
 */

export function useBuilderQueryStates() {
  const [raw, setRaw] = useQueryStates({
    flowID: parseAsString,
    flowVersion: parseAsInteger,
    flowExecutionID: parseAsString,
  });

  const flowID = useMemo(() => parseGraphID(raw.flowID), [raw.flowID]);
  const flowVersion = raw.flowVersion;
  const flowExecutionID = useMemo(
    () => parseGraphExecutionID(raw.flowExecutionID),
    [raw.flowExecutionID],
  );

  function setBuilderQueryStates(patch: {
    flowID?: GraphID | string | null;
    flowVersion?: number | null;
    flowExecutionID?: GraphExecutionID | string | null;
  }) {
    setRaw({
      flowID: (patch.flowID as string | null) ?? null,
      flowVersion: patch.flowVersion ?? null,
      flowExecutionID: (patch.flowExecutionID as string | null) ?? null,
    } as never);
  }

  return [
    { flowID, flowVersion, flowExecutionID, raw },
    setBuilderQueryStates,
  ] as const;
}
