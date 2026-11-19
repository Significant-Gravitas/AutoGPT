import type { Expert } from "@/app/api/__generated__/models/expert";
import { NumberTicker } from "@/components/atoms/NumberTicker/NumberTicker";
import { Text } from "@/components/atoms/Text/Text";
import { AnimatePresence, motion } from "framer-motion";
import { ownerOf, type BoardColumn } from "../../../../helpers";
import { BoardCard } from "../BoardCard/BoardCard";

interface Props {
  column: BoardColumn;
  experts: Expert[];
  selectedID: string | null;
  onOpen: (id: string) => void;
}

/** One lane of the board, as tall as the board itself; its cards scroll
 *  inside it. It is never shorter than a few cards, so a short screen
 *  still shows a lane. */
export function BoardLane({ column, experts, selectedID, onOpen }: Props) {
  const headingID = `task-lane-${column.id}`;

  return (
    <section
      aria-labelledby={headingID}
      data-testid="board-lane"
      className="flex min-h-56 min-w-64 flex-1 basis-0 flex-col rounded-tray bg-surface-tray p-1.5"
    >
      <header className="flex h-9 shrink-0 items-center gap-2 px-2.5">
        <Text as="h2" id={headingID} variant="inbox-medium" tone="ink">
          {column.label}
        </Text>
        <Text as="span" variant="inbox" tone="faint" tabular>
          <NumberTicker value={column.tasks.length} />
        </Text>
      </header>
      <motion.ul
        layoutScroll
        className="-m-0.5 scrollbar-none grid min-h-0 flex-1 content-start gap-1.5 overflow-y-auto overscroll-y-contain p-0.5"
      >
        <AnimatePresence initial={false}>
          {column.tasks.map((task) => (
            <BoardCard
              key={task.id}
              task={task}
              owner={ownerOf(task, experts)}
              selected={selectedID === task.id}
              onOpen={() => onOpen(task.id)}
            />
          ))}
        </AnimatePresence>
        {column.tasks.length === 0 ? (
          <li className="py-5 text-center">
            <Text as="span" variant="inbox" tone="faint">
              Nothing here
            </Text>
          </li>
        ) : null}
      </motion.ul>
    </section>
  );
}
