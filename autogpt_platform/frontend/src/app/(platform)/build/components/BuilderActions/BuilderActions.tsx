import { parseAsString, useQueryStates } from "nuqs";
import { AgentOutputs } from "./components/AgentOutputs/AgentOutputs";
import { RunGraph } from "./components/RunGraph/RunGraph";
import { ScheduleGraph } from "./components/ScheduleGraph/ScheduleGraph";
import { PublishToMarketplace } from "./components/PublishToMarketplace/PublishToMarketplace";
import { memo } from "react";

export const BuilderActions = memo(() => {
  const [{ flowID }] = useQueryStates({
    flowID: parseAsString,
  });
  return (
    <div className="absolute bottom-4 left-[50%] z-[100] flex -translate-x-1/2 items-center gap-4 rounded-full bg-white p-2 px-2 shadow-lg">
      <AgentOutputs flowID={flowID} />
      <RunGraph flowID={flowID} />
      <ScheduleGraph flowID={flowID} />
      <PublishToMarketplace flowID={flowID} />
    </div>
  );
});

BuilderActions.displayName = "BuilderActions";
