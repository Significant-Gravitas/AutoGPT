"use client";

import {
  AlertCircleIcon,
  ArrowDown01Icon,
  CheckmarkCircle02Icon,
  Loading03Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";
import { Text } from "@/components/atoms/Text/Text";
import {
  CollapsibleContent,
  CollapsibleRoot,
  CollapsibleTrigger,
} from "@/components/molecules/Collapsible/Collapsible";
import { formatMetrics } from "../../../../../../data/helpers";
import type {
  TaskActivity,
  ToolCallSummary,
} from "../../../../../../data/types";

type TurnActivity = Extract<TaskActivity, { kind: "turn" }>;

interface Props {
  item: TurnActivity;
}

const TOOL_STATUS = {
  ok: { icon: CheckmarkCircle02Icon, tone: "success", label: "Done" },
  error: { icon: AlertCircleIcon, tone: "error", label: "Failed" },
  running: { icon: Loading03Icon, tone: "info", label: "Running" },
} as const;

export function TurnRow({ item }: Props) {
  return (
    <CollapsibleRoot className="group w-full">
      <CollapsibleTrigger asChild>
        <button
          type="button"
          className="flex w-full items-center gap-2 rounded-md px-2 py-1 text-left transition-colors hover:bg-surface-hover focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none"
        >
          <span className="flex min-w-0 flex-1 flex-wrap items-center gap-x-2">
            <Text variant="small-medium" tone="secondary" as="span">
              {item.actor.name}
            </Text>
            <Text variant="small" tone="muted" as="span" tabular>
              {formatMetrics(item.metrics)}
            </Text>
            <Text variant="small" tone="subtle" as="span">
              <RelativeTime date={item.created_at} />
            </Text>
          </span>
          <Icon
            icon={ArrowDown01Icon}
            tone="muted"
            motion="transform"
            aria-hidden
            className="group-data-[state=open]:rotate-180"
          />
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent className="overflow-hidden">
        <div className="flex flex-col gap-2 pt-2 pl-2">
          {item.summary ? (
            <Text variant="small" tone="body" className="whitespace-pre-wrap">
              {item.summary}
            </Text>
          ) : null}
          {item.tool_calls.length ? (
            item.tool_calls.map((call) => (
              <ToolCallCard key={call.id} call={call} />
            ))
          ) : (
            <Text variant="small" tone="muted">
              No tool calls in this step.
            </Text>
          )}
        </div>
      </CollapsibleContent>
    </CollapsibleRoot>
  );
}

function ToolCallCard({ call }: { call: ToolCallSummary }) {
  const status = TOOL_STATUS[call.status];
  return (
    <div className="flex flex-col gap-1.5 rounded-lg border border-border-base bg-surface p-3">
      <div className="flex items-center gap-2">
        <Icon
          icon={status.icon}
          tone={status.tone}
          spin={call.status === "running"}
          aria-label={status.label}
        />
        <Text variant="small-medium" tone="primary" className="truncate">
          {call.name}
        </Text>
      </div>
      {call.input_preview ? (
        <div className="rounded-md bg-surface-muted px-2 py-1">
          <Text
            variant="small-mono"
            tone="muted"
            className="line-clamp-3 break-all whitespace-pre-wrap"
          >
            {call.input_preview}
          </Text>
        </div>
      ) : null}
      {call.output_preview ? (
        <div className="rounded-md bg-surface-muted px-2 py-1">
          <Text
            variant="small-mono"
            tone="muted"
            className="line-clamp-4 break-all whitespace-pre-wrap"
          >
            {call.output_preview}
          </Text>
        </div>
      ) : null}
    </div>
  );
}
