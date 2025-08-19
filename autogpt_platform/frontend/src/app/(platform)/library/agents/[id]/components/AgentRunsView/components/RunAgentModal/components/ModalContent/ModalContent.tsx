import { ManualRunForm } from "../ManualRunForm/ManualRunForm";
import { ScheduleRunForm } from "../ScheduleRunForm/ScheduleRunForm";
import { AutomaticTriggerForm } from "../AutomaticTriggerForm/AutomaticTriggerForm";
import { ManualTriggerForm } from "../ManualTriggerForm/ManualTriggerForm";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { RunVariant } from "../../useAgentRunModal";

interface ModalContentProps {
  agent?: GraphMeta;
  variant: RunVariant;
  isLoading: boolean;
  onClose: () => void;
}

export function ModalContent({
  agent,
  variant,
  isLoading,
  onClose,
}: ModalContentProps) {
  if (isLoading) {
    return (
      <div className="flex-1 p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-4 w-3/4 rounded bg-neutral-200"></div>
          <div className="h-4 w-1/2 rounded bg-neutral-200"></div>
          <div className="h-32 w-full rounded bg-neutral-100"></div>
        </div>
      </div>
    );
  }

  if (!agent) return null;

  return (
    <div className="flex-1 overflow-y-auto">
      {variant === "manual" && (
        <ManualRunForm agent={agent} onClose={onClose} />
      )}
      {variant === "schedule" && (
        <ScheduleRunForm agent={agent} onClose={onClose} />
      )}
      {variant === "automatic-trigger" && (
        <AutomaticTriggerForm agent={agent} onClose={onClose} />
      )}
      {variant === "manual-trigger" && (
        <ManualTriggerForm agent={agent} onClose={onClose} />
      )}
    </div>
  );
}
