import { GraphID } from "@/lib/autogpt-server-api";
import { useSearchParams } from "next/navigation";
import { useState } from "react";

export interface NewControlPanelProps {
  // flowExecutionID: GraphExecutionID | undefined;
  visualizeBeads?: "no" | "static" | "animate";
}

export const useNewControlPanel = ({
  // flowExecutionID,
  visualizeBeads: _visualizeBeads,
}: NewControlPanelProps) => {
  const [blockMenuSelected, setBlockMenuSelected] = useState<
    "save" | "block" | "search" | ""
  >("");
  const query = useSearchParams();
  const _graphVersion = query.get("flowVersion");
  const _graphVersionParsed = _graphVersion
    ? parseInt(_graphVersion)
    : undefined;

  const _flowID = (query.get("flowID") as GraphID | null) ?? undefined;
  // const {
  //   agentDescription,
  //   setAgentDescription,
  //   saveAgent,
  //   agentName,
  //   setAgentName,
  //   savedAgent,
  //   isSaving,
  //   isRunning,
  //   isStopping,
  // } = useAgentGraph(
  //   flowID,
  //   graphVersion,
  //   flowExecutionID,
  //   visualizeBeads !== "no",
  // );

  return {
    blockMenuSelected,
    setBlockMenuSelected,
    // agentDescription,
    // setAgentDescription,
    // saveAgent,
    // agentName,
    // setAgentName,
    // savedAgent,
    // isSaving,
    // isRunning,
    // isStopping,
  };
};
