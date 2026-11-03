"use client";

import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { useId, useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { RelativeTime } from "@/components/atoms/RelativeTime/RelativeTime";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatMetrics } from "../../../../../../data/helpers";
import type { TaskActivity } from "../../../../../../data/types";
import type { StepStatus } from "../../../../helpers";
import { Fold } from "../../../Fold/Fold";
import { LiveTimer } from "../../../LiveTimer/LiveTimer";
import { StepRow } from "../../../StepRow/StepRow";
import { ToolCallRow } from "../ToolCallRow/ToolCallRow";

type TurnActivity = Extract<TaskActivity, { kind: "turn" }>;

interface Props {
  item: TurnActivity;
}

function getTurnStatus(item: TurnActivity): StepStatus {
  if (item.tool_calls.some((call) => call.status === "running"))
    return "running";
  if (item.tool_calls.some((call) => call.status === "error")) return "error";
  return "completed";
}

/** A turn of work, styled like a tool-call card: one line that folds open. */
export function TurnRow({ item }: Props) {
  const [open, setOpen] = useState(false);
  const bodyId = useId();
  const status = getTurnStatus(item);

  return (
    <div className="overflow-hidden rounded-xl border border-border-card bg-surface">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 px-3 py-2.5 text-left transition-[background-color] duration-150 ease-out-expo hover:bg-surface-tray focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none focus-visible:ring-inset"
      >
        <StepRow
          status={status}
          label={
            <>
              <Text as="span" variant="small" weight="medium" tone="ink">
                {item.actor.name}
              </Text>
              <Text as="span" variant="small" tone="quiet" tabular>
                {" "}
                · {formatMetrics(item.metrics)}
              </Text>
            </>
          }
          trailing={
            status === "running" ? (
              <LiveTimer since={item.created_at} />
            ) : (
              <RelativeTime date={item.created_at} />
            )
          }
        />
        <Icon
          icon={ArrowDown01Icon}
          tone="faint"
          size={14}
          motion="transform"
          aria-hidden
          className={cn(open && "rotate-180")}
        />
      </button>
      <Fold open={open} id={bodyId}>
        <div className="flex flex-col border-t border-border-hairline">
          {item.summary ? (
            <Text
              variant="small"
              tone="text"
              className="px-3 py-2.5 whitespace-pre-wrap"
            >
              {item.summary}
            </Text>
          ) : null}
          {item.tool_calls.length ? (
            <div className="flex flex-col divide-y divide-border-hairline border-t border-border-hairline first:border-t-0">
              {item.tool_calls.map((call) => (
                <ToolCallRow
                  key={call.id}
                  call={call}
                  since={item.created_at}
                />
              ))}
            </div>
          ) : (
            <Text variant="small" tone="quiet" className="px-3 py-2.5">
              No tool calls in this step.
            </Text>
          )}
        </div>
      </Fold>
    </div>
  );
}
