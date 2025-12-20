import { Button } from "@/components/atoms/Button/Button";
import { GearIcon } from "@phosphor-icons/react";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useAgentSafeMode } from "@/hooks/useAgentSafeMode";

interface Props {
  agent: LibraryAgent;
  onSelectSettings: () => void;
}

export function AgentSettingsButton({ agent, onSelectSettings }: Props) {
  const { hasHITLBlocks } = useAgentSafeMode(agent);

  if (!hasHITLBlocks) {
    return null;
  }

  return (
    <Button
      variant="ghost"
      size="small"
      className="m-0 min-w-0 rounded-full p-0 px-1"
      onClick={onSelectSettings}
      aria-label="Agent Settings"
    >
      <GearIcon size={18} className="text-zinc-600" />
    </Button>
  );
}
