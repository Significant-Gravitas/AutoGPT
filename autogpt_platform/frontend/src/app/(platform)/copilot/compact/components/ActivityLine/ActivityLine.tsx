import { Icon } from "@/components/atoms/Icon/Icon";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { Collapsible } from "@/components/molecules/Collapsible/Collapsible";
import { Alert02Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import type { MessagePart } from "../../../components/ChatMessagesContainer/helpers";
import { summarizeActivity } from "../../helpers";

interface Props {
  parts: MessagePart[];
  isStreaming: boolean;
}

export function ActivityLine({ parts, isStreaming }: Props) {
  const { heading, rows, state } = summarizeActivity(parts, isStreaming);

  return (
    <Collapsible
      className="w-fit max-w-full"
      triggerClassName="w-fit max-w-full justify-start rounded-md text-muted-foreground"
      trigger={
        <span className="flex min-w-0 items-center gap-2">
          {state === "running" ? (
            <LoadingSpinner size="small" className="text-zinc-400" />
          ) : (
            <Icon
              icon={state === "error" ? Alert02Icon : Tick02Icon}
              size={16}
              className={state === "error" ? "text-red-500" : "text-zinc-400"}
            />
          )}
          <Text variant="body" as="span" tone="muted" className="truncate">
            {heading}
          </Text>
        </span>
      }
    >
      <ul className="ml-2 flex flex-col gap-1 border-l border-border pl-4">
        {rows.map((row) => (
          <li key={row.key}>
            <Text
              variant="small"
              as="span"
              tone={row.state === "error" ? "danger" : "muted"}
            >
              {row.text}
            </Text>
          </li>
        ))}
      </ul>
    </Collapsible>
  );
}
