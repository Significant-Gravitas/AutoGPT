import { Text } from "@/components/atoms/Text/Text";
import type { Task } from "../../../data/types";

interface Props {
  tools: Task["inbox"]["tools_used"];
}

const MAX_TOOLS = 3;

export function ToolHints({ tools }: Props) {
  if (!tools.length) return null;
  return (
    <span className="inline-flex min-w-0 items-center gap-2">
      {tools.slice(0, MAX_TOOLS).map((tool) => (
        <Text
          key={tool.name}
          as="span"
          variant="small"
          tone="quiet"
          tabular
          className="whitespace-nowrap"
        >
          {tool.title}
          {tool.count > 1 ? (
            <Text as="span" variant="small" tone="faint" tabular>
              {` ×${tool.count}`}
            </Text>
          ) : null}
        </Text>
      ))}
    </span>
  );
}
