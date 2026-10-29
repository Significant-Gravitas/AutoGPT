"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { PlugSocketIcon } from "@hugeicons/core-free-icons";
import { useCopilotModal } from "../../useCopilotModal";
import { ConnectorRow } from "./ConnectorRow";
import { McpConnectorRow } from "./McpConnectorRow";
import type { ConnectorRow as Row, McpConnectorRequest } from "./helpers";

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
    <div className="overflow-hidden rounded-3xl border border-separator bg-surface">
      <div className="flex items-center gap-3 border-b border-separator px-4 py-5">
        <Icon
          icon={PlugSocketIcon}
          size={18}
          className="text-foreground-subtle"
        />
        <div className="min-w-0 flex-1">
          <h3 className="text-base font-medium text-foreground-strong">
            {remaining.length ? "Connect your apps" : "Your apps are connected"}
          </h3>
          <p className="mt-1 text-sm text-foreground-tertiary">
            Choose the accounts Otto should use.
          </p>
        </div>
        <span
          role="status"
          className="shrink-0 rounded-full border border-separator px-2.5 py-1 text-xs text-foreground-tertiary"
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
        <div className="border-t border-separator px-4 py-2.5">
          <span className="text-sm text-foreground-tertiary">
            Still to connect: {remaining.join(", ")}
          </span>
        </div>
      )}
      <div className="space-y-2 border-t border-separator bg-surface-muted px-4 py-3">
        <p className="text-sm text-foreground-tertiary">
          {remaining.length
            ? "Once these apps are connected, Otto will continue."
            : "Connections are ready. Otto can continue setting up your workflow."}
        </p>
        <span className="text-sm text-foreground-tertiary">
          Looking for something else?{" "}
          <button
            type="button"
            onClick={() => openModal("integrations")}
            className="font-medium text-foreground-strong underline underline-offset-2"
          >
            Browse all connectors
          </button>
        </span>
      </div>
    </div>
  );
}
