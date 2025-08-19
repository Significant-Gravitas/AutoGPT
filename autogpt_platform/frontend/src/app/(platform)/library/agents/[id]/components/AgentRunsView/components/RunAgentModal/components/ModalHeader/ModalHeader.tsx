import { Badge } from "@/components/atoms/Badge/Badge";
import { VariantTabs } from "./VariantTabs";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { RunVariant } from "../../useAgentRunModal";

interface ModalHeaderProps {
  agent?: GraphMeta;
  activeVariant: RunVariant;
  onVariantChange: (variant: RunVariant) => void;
  isLoading: boolean;
}

export function ModalHeader({
  agent,
  activeVariant,
  onVariantChange,
  isLoading,
}: ModalHeaderProps) {
  if (isLoading) {
    return (
      <div className="sticky top-0 border-b border-neutral-200 bg-white p-6">
        <div className="flex items-center justify-between">
          <div className="animate-pulse">
            <div className="mb-2 h-6 w-32 rounded bg-neutral-200"></div>
            <div className="h-4 w-48 rounded bg-neutral-100"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!agent) return null;

  return (
    <div className="sticky top-0 border-b border-neutral-200 bg-white p-6">
      <div className="mb-4 flex items-start justify-between">
        <div className="flex-1">
          <div className="mb-2 flex items-center gap-3">
            <Badge variant="info">New run</Badge>
            {agent.has_external_trigger && (
              <Badge variant="info">Marketplace</Badge>
            )}
          </div>
          <h2 className="mb-1 text-xl font-semibold text-neutral-800">
            {agent.name}
          </h2>
          <p className="line-clamp-2 text-sm text-neutral-600">
            {agent.description}
          </p>
        </div>
      </div>

      <VariantTabs
        activeVariant={activeVariant}
        onVariantChange={onVariantChange}
        hasExternalTrigger={agent.has_external_trigger}
      />
    </div>
  );
}
