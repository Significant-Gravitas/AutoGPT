import { AgentOutputs } from "./components/AgentOutputs/AgentOutputs";
import { RunGraph } from "./components/RunGraph/RunGraph";
import { ScheduleGraph } from "./components/ScheduleGraph/ScheduleGraph";

export const BuilderActions = () => {
  return (
    <div className="absolute bottom-4 left-[50%] z-[100] flex -translate-x-1/2 items-center gap-2 gap-4">
      <AgentOutputs />
      <RunGraph />
      <ScheduleGraph />
    </div>
  );
};
