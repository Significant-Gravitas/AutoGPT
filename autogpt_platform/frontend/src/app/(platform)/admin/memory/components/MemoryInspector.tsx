import { Text } from "@/components/atoms/Text/Text";
import { environment } from "@/services/environment";
import { MemoryVisualizer } from "./MemoryVisualizer";

export function MemoryInspector() {
  const isAvailable = environment.isGraphMemoryAvailable();

  return (
    <div className="mx-auto flex flex-col gap-4 p-6">
      <Text variant="h1">Memory Inspector</Text>
      {isAvailable ? (
        <>
          <Text variant="body" tone="secondary">
            View entities, facts, and communities stored in your memory graph.
            Trigger a community rebuild on demand.
          </Text>
          <MemoryVisualizer />
        </>
      ) : (
        <Text variant="body" tone="secondary">
          Long-term memory is unavailable on this installation.
        </Text>
      )}
    </div>
  );
}
