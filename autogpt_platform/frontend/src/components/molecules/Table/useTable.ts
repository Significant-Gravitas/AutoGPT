import { useState, useEffect, useRef } from "react";

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

  const prevDefaultValuesRef = useRef<string>();

  useEffect(() => {
    // Serialize current defaultValues for comparison
    const currentStringified = JSON.stringify(defaultValues);

    // Only update if the serialized values actually changed
    if (prevDefaultValuesRef.current !== currentStringified) {
      prevDefaultValuesRef.current = currentStringified;

      if (defaultValues && defaultValues.length > 0) {
        setRows(defaultValues);
      } else if (!defaultValues || defaultValues.length === 0) {
        setRows((prevRows) => {
          // Only clear if we had rows before
          if (prevRows.length > 0) {
            return [];
          }
          return prevRows;
        });
      }
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
