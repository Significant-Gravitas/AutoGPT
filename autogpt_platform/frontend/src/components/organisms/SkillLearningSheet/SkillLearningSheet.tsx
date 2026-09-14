"use client";

import { ExpertSidePanel } from "@/app/(platform)/team/components/ExpertSidePanel/ExpertSidePanel";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { canRestore } from "@/services/skill-learning/helpers";
import { ChangesView } from "./components/ChangesView";
import { DecisionView } from "./components/DecisionView";
import { EditSkillForm } from "./components/EditSkillForm";
import { HistoryView } from "./components/HistoryView";
import { SourcesView } from "./components/SourcesView";
import { SummaryView } from "./components/SummaryView";
import { useSkillLearningSheet } from "./useSkillLearningSheet";
import {
  SheetView,
  SkillReviewState,
  UpdateSkillReview,
  useSkillReviewState,
} from "./useSkillReviewState";

export interface LearningScope {
  /** ``null`` is the personal (AutoPilot) scope. */
  expertId: string | null;
  name: string;
  avatarUrl: string | null;
  color?: string | null;
}

interface Props {
  scope: LearningScope;
  skillName: string | null;
  initialVersionId: string | null;
  onChanged: () => void;
  onClose: () => void;
}

const VIEWS: { value: SheetView; label: string }[] = [
  { value: "summary", label: "Summary" },
  { value: "changes", label: "Changes" },
  { value: "sources", label: "Sources" },
  { value: "history", label: "History" },
];

export function SkillLearningSheet({
  scope,
  skillName,
  initialVersionId,
  onChanged,
  onClose,
}: Props) {
  const review = useSkillReviewState(
    scope.expertId,
    skillName,
    initialVersionId,
  );
  return (
    <ExpertSidePanel
      identity={
        skillName
          ? {
              name: scope.name,
              avatarUrl: scope.avatarUrl,
              color: scope.color,
              isAutopilot: scope.expertId === null,
            }
          : null
      }
      title={skillName ?? ""}
      panelId="skill-learning"
      closeLabel="Close skill learning panel"
      onClose={onClose}
    >
      {skillName ? (
        <SheetBody
          key={`${scope.expertId ?? "personal"}:${skillName}:${initialVersionId ?? ""}`}
          expertId={scope.expertId}
          skillName={skillName}
          state={review.state}
          update={review.update}
          restoreScroll={review.restoreScroll}
          rememberScroll={review.rememberScroll}
          onChanged={onChanged}
        />
      ) : null}
    </ExpertSidePanel>
  );
}

interface BodyProps {
  expertId: string | null;
  skillName: string;
  state: SkillReviewState;
  update: UpdateSkillReview;
  restoreScroll: (element: HTMLDivElement | null) => void;
  rememberScroll: (top: number) => void;
  onChanged: () => void;
}

function SheetBody({
  expertId,
  skillName,
  state,
  update,
  restoreScroll,
  rememberScroll,
  onChanged,
}: BodyProps) {
  const sheet = useSkillLearningSheet({
    expertId,
    skillName,
    state,
    update,
    onChanged,
  });
  const editing = sheet.editor !== null;

  if (sheet.isLoading) {
    return (
      <div className="flex flex-col gap-3 p-5">
        <Skeleton className="h-6 w-1/2" />
        <Skeleton className="h-24 w-full" />
      </div>
    );
  }
  if (sheet.isError || !sheet.detail) {
    return (
      <div className="p-5">
        <ErrorCard
          context="this skill's learning history"
          hint="We could not load learning details for this skill."
          onRetry={() => sheet.refetch()}
        />
      </div>
    );
  }
  const detail = sheet.detail;
  const version = sheet.selectedVersion;
  const currentId = detail.current_version?.id ?? null;
  const restorable = version ? canRestore(version, currentId) : false;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div
        className="min-h-0 flex-1 overflow-y-auto px-5 py-4"
        ref={restoreScroll}
        onScroll={(event) => rememberScroll(event.currentTarget.scrollTop)}
      >
        {sheet.editor ? (
          <EditSkillForm
            currentVersion={detail.current_version ?? null}
            draft={sheet.editor}
            isBusy={sheet.isBusy}
            onDraftChange={sheet.updateEditor}
            onReviewCurrent={sheet.reviewCurrent}
            onSave={sheet.saveEdit}
            onCancel={sheet.cancelEdit}
          />
        ) : (
          <TabsLine
            variant="compact"
            value={sheet.view}
            onValueChange={(value) => sheet.setView(value as SheetView)}
          >
            <TabsLineList className="overflow-x-auto">
              {VIEWS.map((item) => (
                <TabsLineTrigger key={item.value} value={item.value}>
                  {item.label}
                </TabsLineTrigger>
              ))}
            </TabsLineList>
            <TabsLineContent value="summary">
              <div className="flex flex-col gap-4 pt-3">
                {detail.open_decision ? (
                  <DecisionView
                    proposal={detail.open_decision}
                    current={detail.current_version ?? null}
                    draft={sheet.decisionDraft}
                    isBusy={sheet.isBusy}
                    onDraftChange={sheet.setDecisionDraft}
                    onDecide={sheet.decideProposal}
                  />
                ) : null}
                <SummaryView
                  detail={detail}
                  version={version}
                  isBusy={sheet.isBusy}
                  onPolicy={sheet.updatePolicy}
                  onEdit={() => {
                    if (version) sheet.startEdit(version);
                  }}
                  onReport={sheet.reportOutcome}
                />
              </div>
            </TabsLineContent>
            <TabsLineContent value="changes">
              <div className="pt-3">
                <ChangesView
                  version={version}
                  versions={detail.versions ?? []}
                  onSelect={sheet.selectVersion}
                />
              </div>
            </TabsLineContent>
            <TabsLineContent value="sources">
              <div className="pt-3">
                <SourcesView
                  version={version}
                  isBusy={sheet.isBusy}
                  onExclude={sheet.excludeSource}
                />
              </div>
            </TabsLineContent>
            <TabsLineContent value="history">
              <div className="pt-3">
                <HistoryView
                  versions={detail.versions ?? []}
                  currentVersionId={currentId}
                  isBusy={sheet.isBusy}
                  onRestore={sheet.restoreVersion}
                  onSelect={(id) => {
                    sheet.selectVersion(id);
                    sheet.setView("summary");
                  }}
                />
              </div>
            </TabsLineContent>
          </TabsLine>
        )}
      </div>
      {!editing && version && version.id !== currentId ? (
        <div className="flex shrink-0 justify-end gap-2 border-t border-t-sidebar-border px-5 py-3">
          <Button
            variant="primary"
            size="small"
            disabled={sheet.isBusy || !restorable}
            onClick={() => sheet.restoreVersion(version.id)}
          >
            Restore v{version.version}
          </Button>
        </div>
      ) : null}
    </div>
  );
}
