import { NumberTicker } from "@/components/atoms/NumberTicker/NumberTicker";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import type { Task } from "../../data/types";
import { buildBoardColumns, type InboxTab } from "../../helpers";
import { InboxRow } from "../InboxRow/InboxRow";
import { useInboxMotion } from "../InboxList/useInboxMotion";

interface Props {
  tasks: Task[];
  tab: InboxTab;
  expandedID: string | null;
  selectedID: string | null;
  onToggle: (id: string) => void;
  onOpen: (id: string) => void;
}

export function InboxBoard({
  tasks,
  tab,
  expandedID,
  selectedID,
  onToggle,
  onOpen,
}: Props) {
  const columns = buildBoardColumns(tasks, tab);

  return (
    <LayoutGroup id="inbox-board">
      <div
        className={cn(
          "grid items-start gap-3",
          columns.length === 4 ? "grid-cols-4" : "grid-cols-3",
        )}
      >
        {columns.map((column) => (
          <section
            key={column.id}
            aria-labelledby={`inbox-column-${column.id}`}
            className="flex flex-col gap-1.5 rounded-3xl bg-surface-tray p-1.5"
          >
            <header className="flex items-center justify-between px-2.5 pt-1.5 pb-0.5">
              <Text
                variant="small"
                weight="medium"
                tone="ink"
                as="h2"
                id={`inbox-column-${column.id}`}
              >
                {column.label}
              </Text>
              <Text variant="small" tone="quiet" as="span">
                <NumberTicker value={column.tasks.length} />
              </Text>
            </header>
            <ul className="flex min-h-12 flex-col gap-1.5">
              <AnimatePresence initial={false}>
                {column.tasks.map((task) => (
                  <BoardItem
                    key={task.id}
                    task={task}
                    expanded={expandedID === task.id}
                    selected={selectedID === task.id}
                    onToggle={() => onToggle(task.id)}
                    onOpen={() => onOpen(task.id)}
                  />
                ))}
              </AnimatePresence>
            </ul>
          </section>
        ))}
      </div>
    </LayoutGroup>
  );
}

interface ItemProps {
  task: Task;
  expanded: boolean;
  selected: boolean;
  onToggle: () => void;
  onOpen: () => void;
}

function BoardItem({ task, expanded, selected, onToggle, onOpen }: ItemProps) {
  const motionProps = useInboxMotion(`inbox-task-${task.id}`);

  return (
    <motion.li {...motionProps}>
      <InboxRow
        task={task}
        expanded={expanded}
        selected={selected}
        onToggle={onToggle}
        onOpen={onOpen}
        variant="board"
      />
    </motion.li>
  );
}
