"use client";

import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { withFeatureFlag } from "@/services/feature-flags/with-feature-flag";
import { useEffect } from "react";

import { EraseMemoryCard } from "./components/EraseMemoryCard";
import { MemoryChatPanel } from "./components/MemoryChatPanel/MemoryChatPanel";
import { useMemoryChatPanel } from "./components/MemoryChatPanel/useMemoryChatPanel";
import { RecentMemoriesCard } from "./components/RecentMemoriesCard";
import { ScopeCard } from "./components/ScopeCard";
import { SummaryCard } from "./components/SummaryCard";
import { useMemoryPage } from "./useMemoryPage";

function SettingsMemoryPage() {
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

  return (
    <div className="flex flex-col gap-4">
      <header className="flex min-w-0 flex-col pb-2 pl-4">
        <Text variant="h4" as="h1" className="leading-[28px] text-textBlack">
          Memory
        </Text>
        <Text variant="body" className="mt-4 max-w-[600px] text-zinc-700">
          What AutoGPT remembers about you — and who remembers it.
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

export default withFeatureFlag(SettingsMemoryPage, "graphiti-memory");
