"use client";

import { summarizeData } from "./commandOutput";
import { stepFailed, stepOutput } from "./stepOutcome";
import { DataPreview } from "./DataPreview";
import type { ChainRow } from "./helpers";
import { asObject, str } from "./resultHelpers";
import { TechnicalDetails } from "./TechnicalDetails";
import { KeyValueList, Terminal } from "./ToolResultViews";

interface Props {
  row: ChainRow;
}

/** A shell step as a business owner reads it: whether it worked, and a
 *  preview of any table or data it produced. The command and its raw output
 *  are for whoever wants them, one click away; approving a call does not open
 *  them (the approval record shows the command approved). */
export function CommandResult({ row }: Props) {
  const output = asObject(row.output);
  const input = asObject(row.input);
  const stdout = output ? str(output, "stdout") : null;
  const data = stdout ? summarizeData(stdout) : null;
  // Read from the text when the result was too large to arrive as JSON.
  const failed = stepFailed(
    input ? str(input, "command") : null,
    stepOutput(row.output),
  );
  // The tool can refuse a call outright (no command, no sandbox): an error
  // payload with no exit code at all.
  const refusal = output?.type === "error" ? str(output, "message") : null;

  return (
    <div className="flex flex-col items-start gap-1.5">
      {failed && (
        <p className="text-sm text-error">This step didn&apos;t work</p>
      )}
      {data && stdout && <DataPreview summary={data} text={stdout} />}
      <TechnicalDetails>
        <Terminal row={row} />
        {refusal && <p className="text-xs text-error">{refusal}</p>}
        {/* Output too large for the chat arrives as a preview envelope: a
            string that is not the JSON response the terminal reads. */}
        {typeof row.output === "string" && !output && (
          <KeyValueList value={row.output} />
        )}
      </TechnicalDetails>
    </div>
  );
}
