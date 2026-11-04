"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { InsetSidePanel } from "@/components/layout/InsetSidePanel/InsetSidePanel";
import { SegmentedControl } from "@/components/molecules/SegmentedControl/SegmentedControl";
import type { RecipientOption } from "../../../ChatInput/components/RecipientChip";
import {
  ConnectionsGroup,
  InfoHeader,
  ScheduledGroup,
  SkillsGroup,
  TasksGroup,
  WorkflowsGroup,
  WorkInfoGroup,
} from "./components/InfoSections";
import { getExpertRole } from "./helpers";
import { useExpertInfoPanel, type InfoTab } from "./useExpertInfoPanel";

const TABS: { value: InfoTab; label: string }[] = [
  { value: "details", label: "Details" },
  { value: "access", label: "Access" },
  { value: "tasks", label: "Tasks" },
];

const OTTO_DESCRIPTION =
  "Your personal Head of AI. Tell Otto about your work and it finds what to automate, runs subtasks and brings in experts when a job needs them.";

interface Props {
  recipient: RecipientOption;
  isOpen: boolean;
  onClose: () => void;
}

export function ExpertInfoPanel({ recipient, isOpen, onClose }: Props) {
  const panel = useExpertInfoPanel(recipient.id, isOpen);
  const isOtto = recipient.id === null;

  return (
    <InsetSidePanel
      isOpen={isOpen}
      title="Profile"
      ariaLabel={`About ${recipient.name}`}
      closeLabel="Close expert info"
      onClose={onClose}
    >
      <div className="flex flex-col gap-6 px-1.5 pb-6">
        {isOtto ? (
          <>
            <InfoHeader
              recipient={recipient}
              role="Head of AI"
              description={OTTO_DESCRIPTION}
            />
            <ScheduledGroup
              schedules={panel.schedules}
              isPaused={false}
              isLoading={panel.isLoadingSchedules}
            />
            <TasksGroup
              tasks={panel.tasks}
              isLoading={panel.isLoadingTasks}
              isTeam
            />
          </>
        ) : panel.isLoading ? (
          <div className="flex justify-center py-10">
            <LoadingSpinner size="small" />
          </div>
        ) : panel.expert ? (
          <ExpertDetails
            recipient={recipient}
            expert={panel.expert}
            panel={panel}
          />
        ) : (
          <Text variant="body" tone="muted">
            This expert&apos;s profile couldn&apos;t be loaded.
          </Text>
        )}
      </div>
    </InsetSidePanel>
  );
}

interface DetailsProps {
  recipient: RecipientOption;
  expert: Expert;
  panel: ReturnType<typeof useExpertInfoPanel>;
}

function ExpertDetails({ recipient, expert, panel }: DetailsProps) {
  return (
    <>
      <InfoHeader
        recipient={recipient}
        role={getExpertRole(expert)}
        description={expert.tagline || expert.bio}
      />
      <SegmentedControl<InfoTab>
        value={panel.tab}
        onChange={panel.setTab}
        options={TABS}
        ariaLabel={`${expert.name} details`}
      />
      {panel.tab === "details" && (
        <>
          <ScheduledGroup
            schedules={panel.schedules}
            isPaused={Boolean(expert.schedules_paused_at)}
            isLoading={panel.isLoadingSchedules}
          />
          <WorkInfoGroup expert={expert} />
        </>
      )}
      {panel.tab === "access" && (
        <>
          <ConnectionsGroup
            credentials={panel.credentials}
            isLoading={panel.isLoadingCredentials}
          />
          <SkillsGroup skills={expert.skills} />
          <WorkflowsGroup workflows={expert.workflows} />
        </>
      )}
      {panel.tab === "tasks" && (
        <TasksGroup tasks={panel.tasks} isLoading={panel.isLoadingTasks} />
      )}
    </>
  );
}
