import type { Expert } from "@/app/api/__generated__/models/expert";
import { SkillLearningSheet } from "@/components/organisms/SkillLearningSheet/SkillLearningSheet";
import { learningSheetScope } from "../helpers";
import { useLearningHistory } from "../useLearningHistory";
import { OpenDecisionsCard } from "./OpenDecisionsCard";
import { LearningHistoryCard } from "./LearningHistoryCard";

interface Props {
  learning: ReturnType<typeof useLearningHistory>;
  experts: Expert[];
}

export function MemoryLearningSection({ learning, experts }: Props) {
  if (!learning.enabled) return null;
  const record = learning.openRecord;
  const canOpen =
    !record?.expertId ||
    experts.some((expert) => expert.id === record.expertId);
  return (
    <>
      <OpenDecisionsCard
        decisions={learning.decisions}
        experts={experts}
        isDeciding={learning.isDeciding}
        onDecide={learning.decideProposal}
      />
      <LearningHistoryCard
        items={learning.items}
        isLoading={learning.isLoading}
        experts={experts}
        origin={learning.origin}
        state={learning.state}
        onOriginChange={learning.setOrigin}
        onStateChange={learning.setState}
        onOpen={learning.openRecordFor}
      />
      <SkillLearningSheet
        scope={learningSheetScope(
          learning.openRecord?.expertId ?? null,
          experts,
        )}
        skillName={canOpen ? (record?.skillName ?? null) : null}
        initialVersionId={learning.openRecord?.versionId ?? null}
        onChanged={learning.refreshHistory}
        onClose={learning.closeRecord}
      />
    </>
  );
}
