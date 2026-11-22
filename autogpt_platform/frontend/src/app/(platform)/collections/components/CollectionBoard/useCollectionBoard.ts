import type { PageCollectionBoard } from "@/app/api/__generated__/models/pageCollectionBoard";
import { useState } from "react";
import type { TableRow } from "../CollectionTable/helpers";
import { useRowWrites } from "../CollectionTable/useRowWrites";
import { useRowButtons } from "../RowButtons/useRowButtons";

export const NO_VALUE = "No value";

interface Args {
  section: PageCollectionBoard;
  canEdit: boolean;
  onRunStarted?(libraryAgentID: string, runID: string): void;
}

export function useCollectionBoard({ section, canEdit, onRunStarted }: Args) {
  const writes = useRowWrites(section.collection_id);
  const buttons = useRowButtons({
    onRunStarted,
    onRunFinished: () => void writes.refresh(),
  });
  const [dragged, setDragged] = useState<string | null>(null);
  const [dropTarget, setDropTarget] = useState<string | null>(null);
  const readOnly = section.read_only || !canEdit;
  const rows = section.columns.flatMap((column) => column.rows);

  /** Dragging a card to another column is an edit of the field it groups by. */
  function move(row: TableRow, option: string | null) {
    if ((row.values[section.group_by] ?? null) === option) return;
    void writes.saveCell(row, section.group_by, option);
  }

  return {
    readOnly,
    buttons,
    dropTarget,
    move,
    onDragStart(rowID: string) {
      setDragged(rowID);
    },
    onDragEnd() {
      setDragged(null);
      setDropTarget(null);
    },
    onDragOver(event: React.DragEvent, column: string) {
      if (readOnly || !dragged) return;
      event.preventDefault();
      setDropTarget(column);
    },
    onDragLeave(column: string) {
      setDropTarget((current) => (current === column ? null : current));
    },
    onDrop(option: string | null) {
      const row = rows.find((r) => r.id === dragged);
      setDragged(null);
      setDropTarget(null);
      if (row && !readOnly) move(row, option);
    },
  };
}
