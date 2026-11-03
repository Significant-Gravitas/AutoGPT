import { LayoutGroup } from "framer-motion";
import { BOARD_COLUMNS, HIDDEN_COLUMNS } from "../../data/helpers";
import type { Task } from "../../data/types";
import { groupByStatuses } from "../../helpers";
import { BoardColumn } from "../BoardColumn/BoardColumn";

interface Props {
  tasks: Task[];
  isLoading: boolean;
  showHidden: boolean;
  onOpenTask: (id: string) => void;
}

export function BoardView({ tasks, isLoading, showHidden, onOpenTask }: Props) {
  const columns = groupByStatuses(
    tasks,
    showHidden
      ? [HIDDEN_COLUMNS[0], ...BOARD_COLUMNS, HIDDEN_COLUMNS[1]]
      : BOARD_COLUMNS,
  );

  return (
    <LayoutGroup>
      <div className="flex w-full items-start gap-3 overflow-x-auto pb-2">
        {columns.map((column) => (
          <BoardColumn
            key={column.id}
            id={column.id}
            label={column.label}
            tasks={column.tasks}
            isLoading={isLoading}
            onOpenTask={onOpenTask}
          />
        ))}
      </div>
    </LayoutGroup>
  );
}
