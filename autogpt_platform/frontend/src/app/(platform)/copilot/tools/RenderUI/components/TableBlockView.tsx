import type { TableBlock } from "@/app/api/__generated__/models/tableBlock";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { getNumericColumns } from "../helpers";
import { BlockCard } from "./BlockCard";

interface Props {
  block: TableBlock;
}

export function TableBlockView({ block }: Props) {
  const numeric = getNumericColumns(block);

  return (
    <BlockCard title={block.title}>
      <div
        data-testid="render-ui-table-scroll"
        className="overflow-x-auto rounded-xl bg-surface smooth-shadow-ring-sm"
      >
        <table className="w-full border-collapse text-left">
          <thead className="bg-gradient-to-b from-surface-muted to-surface-hover/50">
            <tr>
              {block.columns.map((column, index) => (
                <th
                  key={index}
                  scope="col"
                  className={cn(
                    "px-4 py-2.5 font-normal whitespace-nowrap",
                    numeric[index] && "text-right",
                  )}
                >
                  <Text as="span" variant="small-medium" tone="secondary">
                    {column}
                  </Text>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {block.rows.map((row, rowIndex) => (
              <tr
                key={rowIndex}
                className="border-t border-separator transition-colors hover:bg-surface-muted/60"
              >
                {row.map((cell, cellIndex) => (
                  <td
                    key={cellIndex}
                    className={cn(
                      "px-4 py-2.5 align-top",
                      // The first column wraps; the rest stay on one line,
                      // so a narrow screen scrolls the table sideways.
                      cellIndex === 0 ? "min-w-44" : "whitespace-nowrap",
                      numeric[cellIndex] && "text-right",
                    )}
                  >
                    <Text
                      as="span"
                      variant={cellIndex === 0 ? "body-medium" : "body"}
                      tone={cellIndex === 0 ? "primary" : "body"}
                      tabular={numeric[cellIndex]}
                      className="break-words"
                    >
                      {cell}
                    </Text>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </BlockCard>
  );
}
