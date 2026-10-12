import type { BoardColumn as BoardColumnModel } from "@/app/api/__generated__/models/boardColumn";
import { Text } from "@/components/atoms/Text/Text";
import { useDroppable } from "@dnd-kit/core";
import {
  SortableContext,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import type { ReactNode } from "react";
import type { Card } from "../../board-helpers";
import { AddTaskCard } from "../AddTaskCard/AddTaskCard";
import { ColumnNameInput } from "../ColumnEditor/ColumnNameInput";
import { SortableTaskCard } from "../SortableTaskCard/SortableTaskCard";

interface Props {
  column: BoardColumnModel;
  cards: Card[];
  onOpenCard: (sessionId: string) => void;
  autoExpandAdd?: boolean;
  isRenaming?: boolean;
  onRename?: (name: string) => void;
  onCancelRename?: () => void;
  onDismissRename?: () => void;
  menu?: ReactNode;
  menuRef?: (element: HTMLElement | null) => void;
  footer?: ReactNode;
}

export function BoardColumn({
  column,
  cards,
  onOpenCard,
  autoExpandAdd = false,
  isRenaming = false,
  onRename = () => {},
  onCancelRename = () => {},
  onDismissRename = () => {},
  menu,
  menuRef,
  footer,
}: Props) {
  const { setNodeRef } = useDroppable({ id: column.key });

  return (
    <section
      ref={setNodeRef}
      aria-label={column.name}
      className="flex max-h-full w-72 shrink-0 flex-col rounded-xl bg-zinc-100/80"
    >
      <header className="flex h-10 shrink-0 items-center gap-2 px-3">
        {isRenaming ? (
          <ColumnNameInput
            columnKey={column.key}
            name={column.name}
            onSave={onRename}
            onCancel={onCancelRename}
            onDismiss={onDismissRename}
          />
        ) : (
          <>
            <Text
              variant="body-medium"
              as="h2"
              className="truncate text-zinc-800"
            >
              {column.name}
            </Text>
            <Text
              variant="small"
              tone="muted"
              as="span"
              className="tabular-nums"
            >
              {cards.length}
            </Text>
            {menu && (
              <div ref={menuRef} className="ml-auto">
                {menu}
              </div>
            )}
          </>
        )}
      </header>
      <div className="flex min-h-0 flex-col gap-2 overflow-y-auto px-2 pb-2">
        <AddTaskCard columnKey={column.key} autoExpand={autoExpandAdd} />
        <SortableContext
          items={cards.map((card) => card.id)}
          strategy={verticalListSortingStrategy}
        >
          {cards.map((card) => (
            <SortableTaskCard key={card.id} card={card} onOpen={onOpenCard} />
          ))}
        </SortableContext>
        {footer}
      </div>
    </section>
  );
}
