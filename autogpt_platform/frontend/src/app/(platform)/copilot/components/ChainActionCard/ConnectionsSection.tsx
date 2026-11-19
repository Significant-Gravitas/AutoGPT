"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { PlugSocketIcon } from "@hugeicons/core-free-icons";
import { useCopilotModal } from "../../useCopilotModal";
import { ConnectorRow } from "./ConnectorRow";
import { McpConnectorRow } from "./McpConnectorRow";
import type { ConnectorRow as Row, McpConnectorRequest } from "./helpers";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  rows: Row[];
  mcp: McpConnectorRequest[];
}

export function ConnectionsSection({ rows, mcp }: Props) {
  const { openModal } = useCopilotModal();
  const mcpRows = mcp.filter(
    (request, i) =>
      mcp.findIndex((other) => other.serverUrl === request.serverUrl) === i,
  );
  const remaining = [
    ...rows
      .filter((row) => !row.selected || row.hasUnansweredTarget)
      .map((row) => row.displayName),
    ...mcpRows
      .filter((request) => !request.connected)
      .map((request) => request.service),
  ];
  return (
    <div className="overflow-hidden rounded-card bg-surface shadow-(--inbox-shadow-card)">
      <div className="flex items-center gap-2.5 border-b border-border-hairline px-3 py-2">
        <Icon icon={PlugSocketIcon} size={15} tone="quiet" />
        <div className="flex min-w-0 flex-1 flex-wrap items-baseline gap-x-2">
          <Text variant="inbox-medium" as="h3" tone="ink">
            {remaining.length ? "Connect your apps" : "Your apps are connected"}
          </Text>
          <Text variant="small" tone="quiet">
            Choose the accounts Otto should use.
          </Text>
        </div>
        <span
          role="status"
          className="shrink-0 text-caption text-foreground-quiet tabular-nums"
        >
          {rows.length + mcpRows.length - remaining.length} of{" "}
          {rows.length + mcpRows.length} ready
        </span>
      </div>
      {rows.map((row) => (
        <ConnectorRow key={row.key ?? row.provider} row={row} />
      ))}
      {mcpRows.map((request) => (
        <McpConnectorRow key={request.id} request={request} />
      ))}
      {remaining.length > 0 && rows.length + mcpRows.length > 1 && (
        <div className="border-t border-border-hairline px-3 py-1.5">
          <Text variant="small" as="span" tone="quiet">
            Still to connect: {remaining.join(", ")}
          </Text>
        </div>
      )}
      <div className="flex flex-wrap gap-x-2 border-t border-border-hairline bg-surface-tray px-3 py-2">
        <Text variant="small" as="span" tone="quiet">
          {remaining.length
            ? "Once these apps are connected, Otto will continue."
            : "Connections are ready. Otto can continue setting up your workflow."}
        </Text>
        <Text variant="small" as="span" tone="quiet">
          Looking for something else?{" "}
          <button
            type="button"
            onClick={() => openModal("integrations")}
            className="rounded-sm font-medium text-foreground-ink underline underline-offset-2 outline-none focus-visible:ring-2 focus-visible:ring-focus-inbox"
          >
            Browse all connectors
          </button>
        </Text>
      </div>
    </div>
  );
}
