import { Text } from "@/components/atoms/Text/Text";
import { useMemoryScope } from "../useMemoryScope";
import { useLearningHistory } from "../useLearningHistory";
import { MemoryLearningSection } from "./MemoryLearningSection";
import { ScopeCard } from "./ScopeCard";
import { UnderstandingCard } from "./UnderstandingCard/UnderstandingCard";

export function UnavailableMemorySettings() {
  const { scopeExpertID, selectScope, experts } = useMemoryScope();
  const learning = useLearningHistory(scopeExpertID);

  return (
    <div className="flex flex-col gap-4">
      <header className="flex min-w-0 flex-col pb-2 pl-4">
        <Text variant="h4" as="h1">
          Memory
        </Text>
        <Text variant="body" tone="body" className="mt-4 max-w-[600px]">
          Long-term memory is unavailable on this installation.
        </Text>
      </header>
      <UnderstandingCard />
      {learning.enabled ? (
        <ScopeCard
          scopeExpertID={scopeExpertID}
          experts={experts}
          onSelect={selectScope}
        />
      ) : null}
      <MemoryLearningSection learning={learning} experts={experts} />
    </div>
  );
}
