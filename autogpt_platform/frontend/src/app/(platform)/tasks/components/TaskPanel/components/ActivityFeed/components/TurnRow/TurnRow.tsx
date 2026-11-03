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
    <div className="overflow-hidden rounded-card bg-surface shadow-card transition-shadow duration-150 ease-out-expo hover:shadow-card-hover">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((value) => !value)}
        className="flex min-h-11 w-full items-center gap-2 px-3.5 text-left transition-[background-color] duration-150 ease-out-expo hover:bg-surface-wash focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none focus-visible:ring-inset"
      >
        <StepRow
          status={status}
          label={
            <>
              <Text as="span" variant="inbox-medium" tone="ink">
                {item.actor.name}
              </Text>
              <Text as="span" variant="inbox" tone="quiet" tabular>
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
        <div className="flex flex-col gap-3 border-t border-border-hairline px-3.5 pt-3 pb-3.5">
          {item.summary ? (
            <Text variant="inbox" tone="text" className="whitespace-pre-wrap">
              {item.summary}
            </Text>
          ) : null}
          {item.tool_calls.length ? (
            <div className="flex flex-col gap-1.5">
              {item.tool_calls.map((call) => (
                <ToolCallRow
                  key={call.id}
                  call={call}
                  since={item.created_at}
                />
              ))}
            </div>
          ) : (
            <Text variant="inbox" tone="quiet">
              No tool calls in this step.
            </Text>
          )}
        </div>
      </Fold>
    </div>
  );
}
