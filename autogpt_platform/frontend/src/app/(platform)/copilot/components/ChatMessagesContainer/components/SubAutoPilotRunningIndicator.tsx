import { ScaleLoader } from "../../ScaleLoader/ScaleLoader";

export function SubAutoPilotRunningIndicator() {
  return (
    <span className="inline-flex items-center gap-1.5 text-neutral-500">
      <ScaleLoader size={16} />
      <span className="animate-pulse [animation-duration:1.5s]">
        Sub-AutoPilot still running&hellip;
      </span>
    </span>
  );
}
