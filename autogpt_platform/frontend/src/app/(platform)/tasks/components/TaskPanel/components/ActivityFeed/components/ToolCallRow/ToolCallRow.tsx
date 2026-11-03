"use client";

import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { useId, useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { ToolCallSummary } from "../../../../../../data/types";
import { Fold } from "../../../Fold/Fold";
import { LiveTimer } from "../../../LiveTimer/LiveTimer";
import { StepRow } from "../../../StepRow/StepRow";

interface Props {
  call: ToolCallSummary;
  since: string;
}

const STATUS = { ok: "completed", error: "error", running: "running" } as const;

/** A tool call in one line; opens to its input and result. */
export function ToolCallRow({ call, since }: Props) {
  const [open, setOpen] = useState(false);
  const bodyId = useId();
  const hasDetails = !!(call.input_preview || call.output_preview);
  const row = (
    <StepRow
      status={STATUS[call.status]}
      label={call.name}
      mono
      trailing={
        call.status === "running" ? <LiveTimer since={since} /> : undefined
      }
    />
  );

  if (!hasDetails) {
    return <div className="flex px-3 py-2">{row}</div>;
  }

  return (
    <div>
      <button
        type="button"
        aria-expanded={open}
        aria-controls={bodyId}
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition-[background-color] duration-150 ease-out-expo hover:bg-surface-tray focus-visible:ring-2 focus-visible:ring-focus focus-visible:outline-none focus-visible:ring-inset"
      >
        {row}
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
        <div className="flex flex-col gap-2 px-3 pb-3 pl-9">
          {call.input_preview ? (
            <Preview label="Input" text={call.input_preview} />
          ) : null}
          {call.output_preview ? (
            <Preview label="Result" text={call.output_preview} />
          ) : null}
        </div>
      </Fold>
    </div>
  );
}

function Preview({ label, text }: { label: string; text: string }) {
  return (
    <div className="flex flex-col gap-1">
      <Text variant="small" tone="faint">
        {label}
      </Text>
      <div className="rounded-lg bg-surface-tray px-2.5 py-1.5">
        <Text
          variant="small-mono"
          tone="text"
          className="line-clamp-6 break-all whitespace-pre-wrap"
        >
          {text}
        </Text>
      </div>
    </div>
  );
}
