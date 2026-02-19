import dynamic from "next/dynamic";

export const FlowRunsTimeline = dynamic(
  () => import("./FlowRunsTimelineChart"),
  { ssr: false },
);

export default FlowRunsTimeline;
