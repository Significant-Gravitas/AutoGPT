"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useEffect } from "react";

import { SkillLearningSheet } from "@/components/organisms/SkillLearningSheet/SkillLearningSheet";
import { EraseMemoryCard } from "./components/EraseMemoryCard";
import { LearningHistoryCard } from "./components/LearningHistoryCard";
import { MemoryChatPanel } from "./components/MemoryChatPanel/MemoryChatPanel";
import { useMemoryChatPanel } from "./components/MemoryChatPanel/useMemoryChatPanel";
import { OpenDecisionsCard } from "./components/OpenDecisionsCard";
import { RecentMemoriesCard } from "./components/RecentMemoriesCard";
import { ScopeCard } from "./components/ScopeCard";
import { SummaryCard } from "./components/SummaryCard";
import { UnderstandingCard } from "./components/UnderstandingCard/UnderstandingCard";
import { learningSheetScope } from "./helpers";
import { useLearningHistory } from "./useLearningHistory";
import { useMemoryPage } from "./useMemoryPage";

export default function SettingsMemoryPage() {
  useEffect(() => {
    document.title = "Memory – AutoGPT Platform";
  }, []);

  const {
    scopeExpertID,
    selectScope,
    experts,
    scopeName,
    facts,
    isLoadingFacts,
    isFactsError,
    factsError,
    refetchFacts,
    memoryCount,
    forgetFact,
    forgettingUuid,
    eraseScope,
    isErasing,
  } = useMemoryPage();

  const chatPanel = useMemoryChatPanel({ scopeExpertID });
  const learning = useLearningHistory(scopeExpertID);

  return (
    <div className="flex flex-col gap-4">
      <header className="flex min-w-0 flex-col pb-2 pl-4">
        <Text variant="h4" as="h1" className="leading-7 text-textBlack">
          Memory
        </Text>
        <Text
          variant="body"
          className="mt-4 max-w-[600px] text-foreground-body"
        >
          {"What AutoGPT remembers about you — and who remembers it."}
        </Text>
      </header>

      <ScopeCard
        scopeExpertID={scopeExpertID}
        experts={experts}
        onSelect={selectScope}
      />

      <SummaryCard
        scopeExpertID={scopeExpertID}
        scopeName={scopeName}
        onViewSummary={() => void chatPanel.openWithSeed("summary")}
      />

      {scopeExpertID === null ? <UnderstandingCard /> : null}

      {isFactsError ? (
        <ErrorCard
          responseError={factsError || undefined}
          context="memory"
          onRetry={refetchFacts}
        />
      ) : (
        <RecentMemoriesCard
          facts={facts}
          isLoading={isLoadingFacts}
          forgettingUuid={forgettingUuid}
          onForget={forgetFact}
          onForgetTopic={() => void chatPanel.openWithSeed("forget")}
        />
      )}

      {learning.enabled ? (
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
            skillName={learning.openRecord?.skillName ?? null}
            initialVersionId={learning.openRecord?.versionId ?? null}
            onChanged={learning.refreshHistory}
            onClose={learning.closeRecord}
          />
        </>
      ) : null}

      <EraseMemoryCard
        scopeName={scopeName}
        memoryCount={memoryCount}
        isErasing={isErasing}
        onErase={eraseScope}
      />

      <MemoryChatPanel
        scopeName={scopeName}
        isOpen={chatPanel.isOpen}
        isStarting={chatPanel.isStarting}
        startError={chatPanel.startError}
        sessionId={chatPanel.sessionId}
        messages={chatPanel.messages}
        status={chatPanel.status}
        error={chatPanel.error}
        queuedMessages={chatPanel.queuedMessages}
        onSend={chatPanel.onSend}
        onStop={chatPanel.stop}
        onRetry={chatPanel.retryStart}
        onClose={chatPanel.close}
      />
    </div>
  );
}
