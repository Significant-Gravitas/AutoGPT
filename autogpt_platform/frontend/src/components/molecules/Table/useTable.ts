import { useState, useEffect } from "react";

export type RowData = Record<string, string>;

interface UseTableOptions {
  columns: string[];
  defaultValues?: RowData[];
  onChange?: (rows: RowData[]) => void;
}

export function useTable({
  columns,
  defaultValues,
  onChange,
}: UseTableOptions) {
  const createEmptyRow = (): RowData => {
    const emptyRow: RowData = {};
    columns.forEach((column) => {
      emptyRow[column] = "";
    });
    return emptyRow;
  };

  const [rows, setRows] = useState<RowData[]>(() => {
    if (defaultValues && defaultValues.length > 0) {
      return defaultValues;
    }
    return [];
  });

  useEffect(() => {
    if (defaultValues !== undefined) {
      setRows(defaultValues);
    }
  }, [defaultValues]);

  const updateRows = (newRows: RowData[]) => {
    setRows(newRows);
    onChange?.(newRows);
  };

  const handleAddRow = () => {
    const newRows = [...rows, createEmptyRow()];
    updateRows(newRows);
  };

  const handleDeleteRow = (rowIndex: number) => {
    const newRows = rows.filter((_, index) => index !== rowIndex);
    updateRows(newRows);
  };

  const handleCellChange = (
    rowIndex: number,
    columnKey: string,
    value: string,
  ) => {
    const newRows = rows.map((row, index) => {
      if (index === rowIndex) {
        return {
          ...row,
          [columnKey]: value,
        };
      }
      return row;
    });
    updateRows(newRows);
  };

  const clearAll = () => {
    updateRows([]);
  };

  return {
    rows,
    handleAddRow,
    handleDeleteRow,
    handleCellChange,
    clearAll,
    createEmptyRow,
  };
}
