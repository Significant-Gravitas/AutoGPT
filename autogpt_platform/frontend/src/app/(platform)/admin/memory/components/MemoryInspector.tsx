import { environment } from "@/services/environment";
import { MemoryVisualizer } from "./MemoryVisualizer";

export function MemoryInspector() {
  const isAvailable = environment.isGraphMemoryAvailable();

  return (
    <div className="mx-auto p-6">
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-3xl font-bold">Memory Inspector</h1>
          <p className="text-foreground-tertiary">
            {isAvailable
              ? "View entities, facts, and communities stored in your Graphiti memory graph. Trigger a community rebuild on demand."
              : "Long-term memory is unavailable on this installation."}
          </p>
        </div>
        {isAvailable ? <MemoryVisualizer /> : null}
      </div>
    </div>
  );
}
