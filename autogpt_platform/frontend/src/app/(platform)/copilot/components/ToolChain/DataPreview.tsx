"use client";

import {
  CodeIcon,
  Download01Icon,
  Table01Icon,
} from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  type DataSummary,
  dataFile,
  describeData,
  previewColumns,
} from "./commandOutput";
import { CARD } from "./ResultCards";

const REVOKE_AFTER_MS = 10_000;

interface Props {
  summary: DataSummary;
  /** What the step printed, as the file the user can download. */
  text: string;
}

/** A table or data a shell step printed: its shape, the first rows or fields,
 *  and the whole of it as a file to download. */
export function DataPreview({ summary, text }: Props) {
  const { title, detail } = describeData(summary);
  const file = dataFile(summary);

  function handleDownload() {
    const url = URL.createObjectURL(new Blob([text], { type: file.mime }));
    const link = document.createElement("a");
    link.href = url;
    link.download = file.name;
    link.click();
    // Revoked once the download has started; revoking at once can cancel it.
    setTimeout(() => URL.revokeObjectURL(url), REVOKE_AFTER_MS);
  }

  return (
    <div className={`${CARD} flex w-full max-w-md flex-col gap-2 p-2.5`}>
      <div className="flex items-center gap-2.5">
        <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-surface-hover">
          <Icon
            icon={summary.kind === "table" ? Table01Icon : CodeIcon}
            size={15}
            className="text-foreground-secondary"
          />
        </div>
        <div className="min-w-0 flex-1">
          <p className="text-xs font-medium text-foreground-base">{title}</p>
          <p className="truncate text-xs text-foreground-subtle">{detail}</p>
        </div>
        <Button
          variant="secondary"
          size="xs"
          leadingIcon={Download01Icon}
          onClick={handleDownload}
        >
          {file.label}
        </Button>
      </div>
      {summary.kind === "table" && <TablePreview summary={summary} />}
      {summary.kind === "json" && summary.sample.length > 0 && (
        <dl className="flex flex-col gap-1 text-xs">
          {summary.sample.map(([key, value]) => (
            <div key={key} className="flex gap-3">
              <dt className="shrink-0 text-foreground-subtle">{key}</dt>
              <dd className="min-w-0 truncate text-foreground-base">{value}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  );
}

interface TablePreviewProps {
  summary: Extract<DataSummary, { kind: "table" }>;
}

function TablePreview({ summary }: TablePreviewProps) {
  const columns = previewColumns(summary.columns, summary.sample);
  if (columns.length === 0) return null;
  const more = summary.rows - summary.sample.length;
  return (
    <div className="overflow-x-auto">
      <table
        aria-label="Table preview"
        className="w-full text-left text-xs text-foreground-base"
      >
        <thead>
          <tr>
            {columns.map((index) => (
              <th
                key={index}
                className="pb-1 pr-3 font-medium text-foreground-subtle"
              >
                {summary.columns[index]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {summary.sample.map((cells, row) => (
            <tr key={row} className="border-t border-border-base/60">
              {columns.map((index) => (
                <td key={index} className="max-w-40 truncate py-1 pr-3">
                  {cells[index]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {more > 0 && (
        <p className="pt-1 text-xs text-foreground-subtle">
          +{more} more row{more === 1 ? "" : "s"}
        </p>
      )}
    </div>
  );
}
