import { NumberTicker } from "@/components/atoms/NumberTicker/NumberTicker";
import { Text } from "@/components/atoms/Text/Text";
import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import type { Task } from "../../data/types";
import { buildBoardColumns } from "../../helpers";
import { InboxRow } from "../InboxRow/InboxRow";
import { useInboxMotion } from "../InboxList/useInboxMotion";

interface Props {
  tasks: Task[];
  selectedID: string | null;
  onOpen: (id: string) => void;
}

export function InboxBoard({ tasks, selectedID, onOpen }: Props) {
  const columns = buildBoardColumns(tasks);

  return (
    <LayoutGroup id="inbox-board">
      <div className="grid h-full grid-cols-3 grid-rows-[minmax(0,1fr)] gap-3 palette-classic">
        {columns.map((column) => (
          <section
            key={column.id}
            aria-labelledby={`inbox-column-${column.id}`}
            className="flex min-h-0 min-w-0 flex-col rounded-tray bg-surface-tray p-1.5"
          >
            <header className="flex items-center gap-1.5 px-3 pt-2 pb-2.5">
              <Text
                variant="inbox-medium"
                tone="ink"
                as="h2"
                id={`inbox-column-${column.id}`}
              >
                {column.label}
              </Text>
              <Text variant="inbox" tone="quiet" as="span" tabular>
                <NumberTicker value={column.tasks.length} />
              </Text>
            </header>
            <motion.ul
              layoutScroll
              className="-mx-1.5 -mt-1 -mb-1.5 flex min-h-12 flex-1 flex-col gap-1.5 overflow-y-auto px-1.5 pt-1 pb-1.5"
            >
              <AnimatePresence initial={false}>
                {column.tasks.map((task) => (
                  <BoardItem
                    key={task.id}
                    task={task}
                    selected={selectedID === task.id}
                    onOpen={() => onOpen(task.id)}
                  />
                ))}
              </AnimatePresence>
            </motion.ul>
          </section>
        ))}
      </div>
    </LayoutGroup>
  );
}

interface ItemProps {
  task: Task;
  selected: boolean;
  onOpen: () => void;
}

function BoardItem({ task, selected, onOpen }: ItemProps) {
  const motionProps = useInboxMotion(`inbox-task-${task.id}`);

  return (
    <motion.li {...motionProps}>
      <InboxRow
        task={task}
        selected={selected}
        onOpen={onOpen}
        variant="board"
      />
    </motion.li>
  );
}
