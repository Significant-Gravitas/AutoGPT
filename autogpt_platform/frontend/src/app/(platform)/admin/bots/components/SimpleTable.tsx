import type { ReactNode } from "react";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  columns: string[];
  rows: ReactNode[][];
  emptyLabel?: string;
}

export function SimpleTable({
  columns,
  rows,
  emptyLabel = "No data yet",
}: Props) {
  if (rows.length === 0) {
    return (
      <Text variant="body" tone="muted" className="py-6 text-center">
        {emptyLabel}
      </Text>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            {columns.map((column, columnIndex) => (
              <th key={columnIndex} className="px-3 py-2 font-medium">
                {column}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex} className="border-b last:border-0">
              {row.map((cell, cellIndex) => (
                <td key={cellIndex} className="px-3 py-2">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
