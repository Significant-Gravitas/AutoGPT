import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";
import { type PageSection, sectionTitle } from "../helpers";
import type { PageEditing, PageLive } from "../Page";
import { CardsSection } from "./CardsSection";
import { ExpertSection } from "./ExpertSection";
import { LatestResultSection } from "./LatestResultSection";
import { NoticeSection } from "./NoticeSection";
import { QuickRunsSection } from "./QuickRunsSection";
import { RecentRunsSection } from "./RecentRunsSection";
import { RunFormSection } from "./RunFormSection";
import { RunHistorySection } from "./RunHistorySection";
import { RunStatsSection } from "./RunStatsSection";
import { ScheduleSection } from "./ScheduleSection";
import { SectionFrame } from "./SectionFrame";
import { WaitingSection } from "./WaitingSection";

interface Props {
  section: PageSection;
  live?: PageLive;
  editing?: PageEditing;
  isFirst: boolean;
  isLast: boolean;
}

export function PageSectionView({
  section,
  live,
  editing,
  isFirst,
  isLast,
}: Props) {
  return (
    <SectionFrame
      index={section.index}
      title={sectionTitle(section)}
      editing={editing}
      isFirst={isFirst}
      isLast={isLast}
      tone={section.kind === "text" ? section.tone : null}
    >
      <SectionBody section={section} live={live} editing={editing} />
    </SectionFrame>
  );
}

function SectionBody({
  section,
  live,
  editing,
}: Pick<Props, "section" | "live" | "editing">): ReactNode {
  switch (section.kind) {
    case "text":
      return (
        <Text variant="body" className="whitespace-pre-line">
          {section.body}
        </Text>
      );
    case "cards":
      return <CardsSection section={section} />;
    case "run_form":
      return <RunFormSection section={section} live={live} />;
    case "quick_runs":
      return <QuickRunsSection section={section} live={live} />;
    case "latest_result":
      return <LatestResultSection section={section} live={live} />;
    case "recent_runs":
      return <RecentRunsSection section={section} />;
    case "run_history":
      return <RunHistorySection section={section} />;
    case "run_numbers":
    case "run_trend":
      return <RunStatsSection section={section} />;
    case "schedule":
      return <ScheduleSection section={section} />;
    case "expert_status":
    case "expert_work":
    case "expert_learned":
    case "expert_activity":
      return <ExpertSection section={section} />;
    case "waiting":
      return <WaitingSection section={section} live={live} />;
    case "notice":
      return <NoticeSection section={section} editing={editing} />;
  }
}
