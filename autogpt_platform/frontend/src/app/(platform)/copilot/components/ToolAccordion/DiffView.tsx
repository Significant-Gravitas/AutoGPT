import { cn } from "@/lib/utils";
import { diffLines } from "diff";

interface DiffRow {
  type: "add" | "del" | "same";
  text: string;
  /** Line number shown in the gutter (new-file number for add/context, old
   *  for removed). */
  no: number;
}

function buildRows(oldText: string, newText: string) {
  const rows: DiffRow[] = [];
  let oldNo = 1;
  let newNo = 1;
  let additions = 0;
  let deletions = 0;

  for (const change of diffLines(oldText, newText)) {
    const type: DiffRow["type"] = change.added
      ? "add"
      : change.removed
        ? "del"
        : "same";
    // jsdiff keeps a trailing newline per hunk — drop the empty tail so we
    // don't render a phantom blank row.
    const lines = change.value.split("\n");
    if (lines[lines.length - 1] === "") lines.pop();

    for (const text of lines) {
      if (type === "add") {
        rows.push({ type, text, no: newNo++ });
        additions++;
      } else if (type === "del") {
        rows.push({ type, text, no: oldNo++ });
        deletions++;
      } else {
        rows.push({ type, text, no: newNo++ });
        oldNo++;
      }
    }
  }

  return { rows, additions, deletions };
}

interface Props {
  oldText: string;
  newText: string;
  /** Full path — the basename is shown in the header. */
  fileName?: string;
}

/** GitHub-style unified line diff: file header with +/- counts, line-number
 *  gutter, and a bold coloured left border on changed lines. */
export function DiffView({ oldText, newText, fileName }: Props) {
  const { rows, additions, deletions } = buildRows(oldText, newText);
  const baseName = fileName?.split("/").filter(Boolean).pop() ?? fileName;

  return (
    <div className="overflow-hidden rounded-lg border border-zinc-200">
      {(baseName || additions > 0 || deletions > 0) && (
        <div className="flex items-center gap-2 border-b border-zinc-200 bg-zinc-50 px-3 py-1.5 text-xs">
          {baseName ? (
            <span className="truncate font-medium text-zinc-700">
              {baseName}
            </span>
          ) : null}
          {additions > 0 ? (
            <span className="font-semibold text-green-600">+{additions}</span>
          ) : null}
          {deletions > 0 ? (
            <span className="font-semibold text-red-500">-{deletions}</span>
          ) : null}
        </div>
      )}

      <div className="overflow-x-auto font-mono text-xs leading-5">
        {rows.map((row, index) => (
          <div
            key={index}
            className={cn(
              "flex",
              row.type === "add" && "bg-green-50",
              row.type === "del" && "bg-red-50",
            )}
          >
            <span
              className={cn(
                "w-[3px] shrink-0",
                row.type === "add" && "bg-green-500",
                row.type === "del" && "bg-red-400",
              )}
            />
            <span
              className={cn(
                "w-10 shrink-0 select-none px-2 text-right tabular-nums",
                row.type === "add"
                  ? "text-green-600"
                  : row.type === "del"
                    ? "text-red-500"
                    : "text-zinc-400",
              )}
            >
              {row.no}
            </span>
            <span
              className={cn(
                "min-w-0 flex-1 whitespace-pre-wrap break-words pr-3",
                row.type === "add"
                  ? "text-green-800"
                  : row.type === "del"
                    ? "text-red-800"
                    : "text-zinc-600",
              )}
            >
              {row.text || " "}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
