import { Badge } from "@/components/atoms/Badge/Badge";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { ShowMoreText } from "@/components/molecules/ShowMoreText/ShowMoreText";
import { ClockIcon } from "lucide-react";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";

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

        {/* Schedule recommendation tip */}
        {agent.recommended_schedule_cron && !agent.has_external_trigger && (
          <div className="mt-4 flex items-center gap-2">
            <ClockIcon className="h-4 w-4 text-gray-500" />
            <p className="text-sm text-gray-600">
              <strong>Tip:</strong> For best results, run this agent{" "}
              {humanizeCronExpression(
                agent.recommended_schedule_cron,
              ).toLowerCase()}
            </p>
          </div>
        )}

        {/* Setup Instructions */}
        {agent.instructions && (
          <div className="mt-4 flex items-start gap-2">
            <svg
              className="mt-0.5 h-4 w-4 flex-shrink-0 text-gray-500"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div className="text-sm text-gray-600">
              <strong>Setup Instructions:</strong>{" "}
              <span className="whitespace-pre-wrap">{agent.instructions}</span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
