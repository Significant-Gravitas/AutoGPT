import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { ResolvedWorkflowPage } from "@/app/api/__generated__/models/resolvedWorkflowPage";
import { Text } from "@/components/atoms/Text/Text";
import { LatestResultSection } from "./LatestResultSection";
import { RecentRunsSection } from "./RecentRunsSection";
import { RunFormSection } from "./RunFormSection";
import { ScheduleSection } from "./ScheduleSection";
import { SectionFrame } from "./SectionFrame";

type Section = ResolvedWorkflowPage["sections"][number];

interface Props {
  section: Section;
  agent?: LibraryAgent;
  onRunStarted?: (run: GraphExecutionMeta) => void;
  onSelectRun?: (runID: string) => void;
  onRemove?: (index: number) => void;
}

export function PageSectionView({
  section,
  agent,
  onRunStarted,
  onSelectRun,
  onRemove,
}: Props) {
  const remove = onRemove ? () => onRemove(section.index) : undefined;

  switch (section.kind) {
    case "text":
      return (
        <SectionFrame title={section.heading} onRemove={remove}>
          <Text variant="body" className="whitespace-pre-line">
            {section.body}
          </Text>
        </SectionFrame>
      );
    case "run_form":
      return (
        <SectionFrame onRemove={remove}>
          <RunFormSection
            section={section}
            agent={agent}
            onRunStarted={onRunStarted}
          />
        </SectionFrame>
      );
    case "latest_result":
      return (
        <SectionFrame title="Latest result" onRemove={remove}>
          <LatestResultSection section={section} />
        </SectionFrame>
      );
    case "recent_runs":
      return (
        <SectionFrame title="Recent runs" onRemove={remove}>
          <RecentRunsSection section={section} onSelectRun={onSelectRun} />
        </SectionFrame>
      );
    case "schedule":
      return (
        <SectionFrame title="Schedule" onRemove={remove}>
          <ScheduleSection section={section} />
        </SectionFrame>
      );
  }
}
