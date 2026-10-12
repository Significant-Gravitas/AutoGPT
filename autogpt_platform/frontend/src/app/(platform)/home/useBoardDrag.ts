import {
  type Announcements,
  type DragEndEvent,
  type DragStartEvent,
  KeyboardSensor,
  MouseSensor,
  TouchSensor,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import { sortableKeyboardCoordinates } from "@dnd-kit/sortable";
import { useQueryClient } from "@tanstack/react-query";
import {
  boardScopeKey,
  type ColumnWithCards,
  dragLabel,
  planDrop,
} from "./board-helpers";
import { useMoveCard } from "./useMoveCard";

// Enter stays free to open the card; Space picks it up.
const KEYBOARD_CODES = {
  start: ["Space"],
  cancel: ["Escape"],
  end: ["Space", "Enter"],
};

const SCREEN_READER_INSTRUCTIONS =
  "To pick up a task, press Space. Use the arrow keys to move it, Space or Enter to drop it, and Escape to cancel.";

interface Args {
  boardRef: string;
  columns: ColumnWithCards[];
  onDraggingChange: (cardId: string | null) => void;
}

export function useBoardDrag({ boardRef, columns, onDraggingChange }: Args) {
  const queryClient = useQueryClient();
  const { move } = useMoveCard(boardRef);

  const sensors = useSensors(
    useSensor(MouseSensor, { activationConstraint: { distance: 6 } }),
    // Press and hold drags; a quick swipe still scrolls the column or board.
    useSensor(TouchSensor, {
      activationConstraint: { delay: 200, tolerance: 5 },
    }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
      keyboardCodes: KEYBOARD_CODES,
    }),
  );

  function label(id: string | number) {
    return dragLabel(columns, String(id));
  }

  const announcements: Announcements = {
    onDragStart: ({ active }) => `Picked up ${label(active.id)}.`,
    onDragOver: ({ active, over }) =>
      over ? `${label(active.id)} is over ${label(over.id)}.` : undefined,
    onDragEnd: ({ active, over }) =>
      over
        ? `Dropped ${label(active.id)} on ${label(over.id)}.`
        : `Dropped ${label(active.id)}.`,
    onDragCancel: ({ active }) => `Put ${label(active.id)} back.`,
  };

  function handleDragStart({ active }: DragStartEvent) {
    onDraggingChange(String(active.id));
    void queryClient.cancelQueries({ queryKey: boardScopeKey(boardRef) });
  }

  function handleDragEnd({ active, over }: DragEndEvent) {
    onDraggingChange(null);
    if (!over) return;
    const dragged = active.rect.current.translated;
    const below =
      !!dragged && dragged.top > over.rect.top + over.rect.height / 2;
    const drop = planDrop(columns, String(active.id), String(over.id), below);
    if (drop) move(String(active.id), drop.column, drop.position);
  }

  return {
    sensors,
    accessibility: {
      announcements,
      screenReaderInstructions: { draggable: SCREEN_READER_INSTRUCTIONS },
    },
    onDragStart: handleDragStart,
    onDragEnd: handleDragEnd,
    onDragCancel: () => onDraggingChange(null),
  };
}
