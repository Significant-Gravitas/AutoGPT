import React, { FC, useCallback, useEffect, useState } from "react";

import { PlusIcon, XIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import NodeHandle from "@/app/(platform)/build/components/legacy-builder/NodeHandle";
import {
  BlockIOTableSubSchema,
  TableRow,
  TableCellValue,
} from "@/lib/autogpt-server-api/types";
import { Input } from "./atoms/Input/Input";
import { Button } from "./atoms/Button/Button";

interface NodeTableInputProps {
  nodeId: string;
  selfKey: string;
  schema: BlockIOTableSubSchema;
  headers: string[];
  rows?: TableRow[];
  errors: { [key: string]: string | undefined };
  connections: {
    edge_id: string;
    source: string;
    sourceHandle: string;
    target: string;
    targetHandle: string;
  }[];
  handleInputChange: (key: string, value: TableRow[]) => void;
  handleInputClick: (key: string) => void;
  className?: string;
  displayName?: string;
}

export const NodeTableInput: FC<NodeTableInputProps> = ({
  nodeId,
  selfKey,
  schema,
  headers,
  rows = [],
  errors,
  connections,
  handleInputChange,
  handleInputClick: _handleInputClick,
  className,
  displayName,
}) => {
  const [tableData, setTableData] = useState<TableRow[]>(rows);

  // Sync with parent state when rows change
  useEffect(() => {
    setTableData(rows);
  }, [rows]);

  const isConnected = (key: string) =>
    connections.some((c) => c.targetHandle === key && c.target === nodeId);

  const updateTableData = useCallback(
    (newData: TableRow[]) => {
      setTableData(newData);
      handleInputChange(selfKey, newData);
    },
    [selfKey, handleInputChange],
  );

  const updateCell = (
    rowIndex: number,
    header: string,
    value: TableCellValue,
  ) => {
    const newData = [...tableData];
    if (!newData[rowIndex]) {
      newData[rowIndex] = {};
    }
    newData[rowIndex][header] = value;
    updateTableData(newData);
  };

  const addRow = () => {
    if (!headers || headers.length === 0) {
      return;
    }
    const newRow: TableRow = {};
    headers.forEach((header) => {
      newRow[header] = "";
    });
    updateTableData([...tableData, newRow]);
  };

  const removeRow = (index: number) => {
    const newData = tableData.filter((_, i) => i !== index);
    updateTableData(newData);
  };

  return (
    <div className={cn("w-full space-y-2", className)}>
      <NodeHandle
        title={displayName || selfKey}
        keyName={selfKey}
        schema={schema}
        isConnected={isConnected(selfKey)}
        isRequired={false}
        side="left"
      />

      {!isConnected(selfKey) && (
        <div className="nodrag overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr>
                {headers.map((header, index) => (
                  <th
                    key={index}
                    className="border border-gray-300 bg-gray-100 px-2 py-1 text-left text-sm font-medium dark:border-gray-600 dark:bg-gray-800"
                  >
                    {header}
                  </th>
                ))}
                <th className="w-10"></th>
              </tr>
            </thead>
            <tbody>
              {tableData.map((row, rowIndex) => (
                <tr key={rowIndex}>
                  {headers.map((header, colIndex) => (
                    <td
                      key={colIndex}
                      className="border border-gray-300 p-1 dark:border-gray-600"
                    >
                      <Input
                        id={`${selfKey}-${rowIndex}-${header}`}
                        label={header}
                        type="text"
                        value={String(row[header] || "")}
                        onChange={(e) =>
                          updateCell(rowIndex, header, e.target.value)
                        }
                        className="h-8 w-full"
                        placeholder={`Enter ${header}`}
                      />
                    </td>
                  ))}
                  <td className="p-1">
                    <Button
                      variant="ghost"
                      size="small"
                      onClick={() => removeRow(rowIndex)}
                      className="h-8 w-8 p-0"
                    >
                      <XIcon />
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          <Button
            className="mt-2 bg-gray-200 font-normal text-black hover:text-white dark:bg-gray-700 dark:text-white dark:hover:bg-gray-600"
            onClick={addRow}
            size="small"
          >
            <PlusIcon className="mr-2" /> Add Row
          </Button>
        </div>
      )}

      {errors[selfKey] && (
        <span className="text-sm text-red-500">{errors[selfKey]}</span>
      )}
    </div>
  );
};
