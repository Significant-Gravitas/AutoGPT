import { Badge } from "@/components/atoms/Badge/Badge";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";

interface ModalHeaderProps {
  showScheduleView: boolean;
  agent: LibraryAgent;
}

export function ModalHeader({ showScheduleView, agent }: ModalHeaderProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Badge variant="info">
          {showScheduleView ? "New Trigger" : "New run"}
        </Badge>
      </div>
      <div>
        <Text variant="h3">{agent.name}</Text>
        <Text variant="body-medium">by {agent.creator_name}</Text>
        <Text variant="small">{agent.description}</Text>
      </div>
    </div>
  );
}
