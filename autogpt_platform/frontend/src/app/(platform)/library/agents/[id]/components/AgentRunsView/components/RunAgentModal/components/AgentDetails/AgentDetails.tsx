import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { Badge } from "@/components/atoms/Badge/Badge";
import { formatAgentStatus, getStatusColor } from "./helpers";
import { formatDate } from "@/lib/utils/time";

interface Props {
  agent: LibraryAgent;
}

export function AgentDetails({ agent }: Props) {
  return (
    <div className="mt-4 flex flex-col gap-5">
      <div>
        <Text variant="body-medium" className="mb-1 !text-black">
          Current Status
        </Text>
        <div className="flex items-center gap-2">
          <div
            className={`h-2 w-2 rounded-full ${getStatusColor(agent.status)}`}
          />
          <Text variant="body" className="!text-zinc-700">
            {formatAgentStatus(agent.status)}
          </Text>
        </div>
      </div>

      <div>
        <Text variant="body-medium" className="mb-1 !text-black">
          Version
        </Text>
        <div className="flex items-center gap-2">
          <Text variant="body" className="!text-zinc-700">
            v{agent.graph_version}
          </Text>
          {agent.is_latest_version && (
            <Badge variant="success" size="small">
              Latest
            </Badge>
          )}
        </div>
      </div>
      <div>
        <Text variant="body-medium" className="mb-1 !text-black">
          Last Updated
        </Text>
        <Text variant="body" className="!text-zinc-700">
          {formatDate(agent.updated_at)}
        </Text>
      </div>
      {agent.has_external_trigger && (
        <div>
          <Text variant="body-medium" className="mb-1">
            Trigger Type
          </Text>
          <Text variant="body" className="!text-neutral-700">
            External Webhook
          </Text>
        </div>
      )}
    </div>
  );
}
