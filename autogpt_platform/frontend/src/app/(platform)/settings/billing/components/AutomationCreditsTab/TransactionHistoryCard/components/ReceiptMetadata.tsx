import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  formatActivityDate,
  statusLabel,
  taskHref,
  type Transaction,
} from "../helpers";

type Props = { transaction: Transaction };

export function ReceiptMetadata({ transaction }: Props) {
  const started = formatActivityDate(transaction.execution_started_at);
  const posted = formatActivityDate(transaction.transaction_time);
  const cutoff = formatActivityDate(transaction.receipt_as_of);
  return (
    <div className="mt-4 flex flex-wrap items-end justify-between gap-4 border-t border-zinc-200 pt-4">
      <div className="flex flex-wrap gap-x-6 gap-y-3">
        {transaction.execution_status && (
          <div className="flex flex-col items-start gap-1">
            <Text variant="small" className="text-zinc-600">
              Run status
            </Text>
            <Badge
              variant={
                transaction.execution_status === "COMPLETED"
                  ? "success"
                  : transaction.execution_status === "FAILED"
                    ? "error"
                    : "info"
              }
              size="small"
            >
              {statusLabel(transaction.execution_status)}
            </Badge>
          </div>
        )}
        {transaction.execution_started_at && (
          <MetadataField
            label="Started"
            value={`${started.date} · ${started.time}`}
            title={started.full}
          />
        )}
        {transaction.execution_graph_version != null && (
          <MetadataField
            label="Version"
            value={`v${transaction.execution_graph_version}`}
          />
        )}
        <MetadataField
          label="Last activity"
          value={`${posted.date} · ${posted.time}`}
          title={posted.full}
        />
        {transaction.receipt_as_of && (
          <MetadataField
            label="Credits as of"
            value={`${cutoff.date} · ${cutoff.time}`}
            title={cutoff.full}
          />
        )}
      </div>
      <ReceiptDestination transaction={transaction} />
    </div>
  );
}

function MetadataField({
  label,
  value,
  title,
}: {
  label: string;
  value: string;
  title?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <Text variant="small" className="text-zinc-600">
        {label}
      </Text>
      <Text variant="small" unmask={false} title={title}>
        {value}
      </Text>
    </div>
  );
}

function ReceiptDestination({ transaction }: Props) {
  const href = taskHref(transaction);
  const conversationHref = transaction.conversation_id
    ? `/copilot?sessionId=${encodeURIComponent(transaction.conversation_id)}`
    : null;
  if (href || conversationHref) {
    return (
      <Button
        as="NextLink"
        href={href || conversationHref || ""}
        variant="secondary"
        size="small"
        className="min-h-11 min-w-0"
        rightIcon={<Icon icon={ArrowUpRight01Icon} size={16} />}
      >
        {href ? "View task" : "View conversation"}
      </Button>
    );
  }
  if (transaction.activity_type === "agent_run") {
    const label = transaction.library_agent_id
      ? "Run unavailable"
      : transaction.execution_available
        ? "Not in your library"
        : "Agent and run unavailable";
    return (
      <Text variant="small" className="text-zinc-600">
        {label}
      </Text>
    );
  }
  if (transaction.activity_type === "copilot_tools")
    return (
      <Text variant="small" className="text-zinc-600">
        Conversation unavailable
      </Text>
    );
  return null;
}
