"use client";

import {
  ArrowDown01Icon,
  CodeIcon,
  Table01Icon,
} from "@hugeicons/core-free-icons";
import { useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { describeData, summarizeData } from "./commandOutput";
import type { ChainRow } from "./helpers";
import { CARD, HALF } from "./ResultCards";
import { asObject, str } from "./resultHelpers";
import { KeyValueList, Terminal } from "./ToolResultViews";

interface Props {
  row: ChainRow;
}

/** A shell step as a business owner reads it: whether it worked, and any
 *  table or data it produced as a chip. The command and its raw output are
 *  for whoever wants them, one click away. */
export function CommandResult({ row }: Props) {
  // A call the user approved shows the command they approved without a click.
  const [showDetails, setShowDetails] = useState(!!row.held);
  const output = asObject(row.output);
  const stdout = output ? str(output, "stdout") : null;
  const data = stdout ? summarizeData(stdout) : null;
  const exitCode = output?.exit_code;
  // The tool can refuse a call outright (no command, no sandbox): an error
  // payload with no exit code at all.
  const refusal = output?.type === "error" ? str(output, "message") : null;
  // A non-zero exit alone is not a failure (`grep -q`, `test -f` and `diff`
  // answer with one); it is when the command also complained on stderr.
  const complained = !!(output && str(output, "stderr"));
  const failed =
    output?.type === "error" ||
    output?.timed_out === true ||
    (typeof exitCode === "number" && exitCode !== 0 && complained);

  return (
    <div className="flex flex-col items-start gap-1.5">
      {failed && (
        <p className="text-sm text-red-500">This step didn&apos;t work</p>
      )}
      {data && (
        <DataChip {...describeData(data)} isTable={data.kind === "table"} />
      )}
      <button
        type="button"
        onClick={() => setShowDetails(!showDetails)}
        aria-expanded={showDetails}
        className="flex items-center gap-1 text-xs text-zinc-400 transition-colors hover:text-zinc-600"
      >
        {showDetails ? "Hide technical details" : "Show technical details"}
        <Icon
          icon={ArrowDown01Icon}
          size={10}
          className={cn(
            "transition-transform duration-300 ease-out-quint",
            showDetails && "rotate-180",
          )}
        />
      </button>
      {showDetails && (
        <div className="flex w-full flex-col gap-1.5">
          <Terminal row={row} />
          {refusal && <p className="text-xs text-red-400">{refusal}</p>}
          {/* Output too large for the chat arrives as a preview envelope. */}
          {typeof row.output === "string" && (
            <KeyValueList value={row.output} />
          )}
        </div>
      )}
    </div>
  );
}

interface DataChipProps {
  title: string;
  detail: string;
  isTable: boolean;
}

function DataChip({ title, detail, isTable }: DataChipProps) {
  return (
    <div className={`${CARD} ${HALF} flex items-center gap-2.5 p-2.5`}>
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
        <Icon
          icon={isTable ? Table01Icon : CodeIcon}
          size={15}
          className="text-zinc-600"
        />
      </div>
      <div className="min-w-0">
        <p className="text-xs font-medium text-zinc-700">{title}</p>
        <p className="truncate text-xs text-zinc-400">{detail}</p>
      </div>
    </div>
  );
}
