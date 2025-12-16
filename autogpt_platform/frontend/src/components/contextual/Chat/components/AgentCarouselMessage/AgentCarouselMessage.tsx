import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowRight, List, Robot } from "@phosphor-icons/react";

export interface Agent {
  id: string;
  name: string;
  description: string;
  version?: number;
}

export interface AgentCarouselMessageProps {
  agents: Agent[];
  totalCount?: number;
  onSelectAgent?: (agentId: string) => void;
  className?: string;
}

export function AgentCarouselMessage({
  agents,
  totalCount,
  onSelectAgent,
  className,
}: AgentCarouselMessageProps) {
  const displayCount = totalCount ?? agents.length;

  return (
    <div
      className={cn(
        "mx-4 my-2 flex flex-col gap-4 rounded-lg border border-purple-200 bg-purple-50 p-6",
        className,
      )}
    >
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-purple-500">
          <List size={24} weight="bold" className="text-white" />
        </div>
        <div>
          <Text variant="h3" className="text-purple-900">
            Found {displayCount} {displayCount === 1 ? "Agent" : "Agents"}
          </Text>
          <Text variant="small" className="text-purple-700">
            Select an agent to view details or run it
          </Text>
        </div>
      </div>

      {/* Agent Cards */}
      <div className="grid gap-3 sm:grid-cols-2">
        {agents.map((agent) => (
          <Card
            key={agent.id}
            className="border border-purple-200 bg-white p-4"
          >
            <div className="flex gap-3">
              <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-lg bg-purple-100">
                <Robot size={20} weight="bold" className="text-purple-600" />
              </div>
              <div className="flex-1 space-y-2">
                <div>
                  <Text
                    variant="body"
                    className="font-semibold text-purple-900"
                  >
                    {agent.name}
                  </Text>
                  {agent.version && (
                    <Text variant="small" className="text-purple-600">
                      v{agent.version}
                    </Text>
                  )}
                </div>
                <Text variant="small" className="line-clamp-2 text-purple-700">
                  {agent.description}
                </Text>
                {onSelectAgent && (
                  <Button
                    onClick={() => onSelectAgent(agent.id)}
                    variant="ghost"
                    className="mt-2 flex items-center gap-1 p-0 text-sm text-purple-600 hover:text-purple-800"
                  >
                    View details
                    <ArrowRight size={16} weight="bold" />
                  </Button>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>

      {totalCount && totalCount > agents.length && (
        <Text variant="small" className="text-center text-purple-600">
          Showing {agents.length} of {totalCount} results
        </Text>
      )}
    </div>
  );
}
