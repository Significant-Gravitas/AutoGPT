import { Badge } from "@/components/atoms/Badge/Badge";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { ShowMoreText } from "@/components/molecules/ShowMoreText/ShowMoreText";

interface ModalHeaderProps {
  agent: LibraryAgent;
}

export function ModalHeader({ agent }: ModalHeaderProps) {
  const isUnknownCreator = agent.creator_name === "Unknown";
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <Badge variant="info">New Run</Badge>
      </div>
      <div>
        <Text variant="h3">{agent.name}</Text>
        {!isUnknownCreator ? (
          <Text variant="body-medium">by {agent.creator_name}</Text>
        ) : null}
        <ShowMoreText
          previewLimit={80}
          variant="small"
          className="mt-4 !text-zinc-700"
        >
          {agent.description}
        </ShowMoreText>
      </div>
    </div>
  );
}
