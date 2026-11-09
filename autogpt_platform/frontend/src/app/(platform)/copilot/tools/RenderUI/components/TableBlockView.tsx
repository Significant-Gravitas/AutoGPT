import type { TableBlock } from "@/app/api/__generated__/models/tableBlock";
import { Text } from "@/components/atoms/Text/Text";
import { BlockCard } from "./BlockCard";

interface Props {
  block: TableBlock;
}

export function TableBlockView({ block }: Props) {
  return (
    <BlockCard title={block.title}>
      <div data-testid="render-ui-table-scroll" className="overflow-x-auto">
        <table className="w-full border-collapse text-left">
          <thead>
            <tr className="border-b border-border-base">
              {block.columns.map((column, index) => (
                <th key={index} className="px-2 py-1.5">
                  <Text
                    variant="small"
                    weight="medium"
                    tone="primary"
                    as="span"
                  >
                    {column}
                  </Text>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {block.rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="border-b border-separator">
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex} className="px-2 py-1.5 align-top">
                    <Text variant="small" tone="body" as="span">
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
