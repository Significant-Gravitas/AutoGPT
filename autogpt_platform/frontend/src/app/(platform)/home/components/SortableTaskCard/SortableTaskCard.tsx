import { useSortable } from "@dnd-kit/sortable";
import type { Card } from "../../board-helpers";
import { TaskCard } from "../TaskCard/TaskCard";

interface Props {
  card: Card;
  onOpen: (sessionId: string) => void;
}

export function SortableTaskCard({ card, onOpen }: Props) {
  const sortable = useSortable({ id: card.id });
  return <TaskCard card={card} onOpen={onOpen} sortable={sortable} />;
}
