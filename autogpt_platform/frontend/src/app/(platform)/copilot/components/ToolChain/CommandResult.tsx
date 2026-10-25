"use client";

import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { stepFailed, summarizeData } from "./commandOutput";
import { DataPreview } from "./DataPreview";
import type { ChainRow } from "./helpers";
import { asObject, str } from "./resultHelpers";
import { KeyValueList, Terminal } from "./ToolResultViews";

interface Props {
  row: ChainRow;
}

/** A shell step as a business owner reads it: whether it worked, and a
 *  preview of any table or data it produced. The command and its raw output
 *  are for whoever wants them, one click away; approving a call does not open
 *  them (the approval record shows the command approved). */
export function CommandResult({ row }: Props) {
  const [showDetails, setShowDetails] = useState(false);
  const output = asObject(row.output);
  const input = asObject(row.input);
  const stdout = output ? str(output, "stdout") : null;
  const data = stdout ? summarizeData(stdout) : null;
  const failed = stepFailed(input ? str(input, "command") : null, output);
  // The tool can refuse a call outright (no command, no sandbox): an error
  // payload with no exit code at all.
  const refusal = output?.type === "error" ? str(output, "message") : null;

  return (
    <div className="flex flex-col items-start gap-1.5">
      {failed && (
        <p className="text-sm text-error">This step didn&apos;t work</p>
      )}
      {data && stdout && <DataPreview summary={data} text={stdout} />}
      <button
        type="button"
        onClick={() => setShowDetails(!showDetails)}
        aria-expanded={showDetails}
        className="flex items-center gap-1 text-xs text-foreground-subtle transition-colors hover:text-foreground-secondary"
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
          {refusal && <p className="text-xs text-error">{refusal}</p>}
          {/* Output too large for the chat arrives as a preview envelope: a
              string that is not the JSON response the terminal reads. */}
          {typeof row.output === "string" && !output && (
            <KeyValueList value={row.output} />
          )}
        </div>
      )}
    </div>
  );
}
