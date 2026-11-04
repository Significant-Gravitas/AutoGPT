import { NumberTicker } from "@/components/atoms/NumberTicker/NumberTicker";
import { AnimatePresence, motion } from "framer-motion";
import type { InboxColumn } from "../../../helpers";
import styles from "../board.module.css";
import { BoardCard } from "./BoardCard/BoardCard";

interface Props {
  column: InboxColumn;
  openID: string | null;
  selectedID: string | null;
  onToggle(id: string): void;
  onOpen(id: string): void;
}

export function BoardLane({
  column,
  openID,
  selectedID,
  onToggle,
  onOpen,
}: Props) {
  const headingID = `inbox-column-${column.id}`;

  return (
    <section aria-labelledby={headingID} className={styles.lane}>
      <header className={styles.laneHead}>
        <span aria-hidden className={styles.laneMark}>
          <svg width="14" height="14" viewBox="0 0 14 14">
            <path
              d="M7 1.5c.5 2.8 1.7 4 4.5 4.5-2.8.5-4 1.7-4.5 4.5-.5-2.8-1.7-4-4.5-4.5 2.8-.5 4-1.7 4.5-4.5Z"
              fill="currentColor"
            />
          </svg>
        </span>
        <h2 id={headingID} className={styles.laneName}>
          {column.label}
        </h2>
        <span className={styles.count}>
          <NumberTicker value={column.tasks.length} />
        </span>
      </header>
      <motion.ul layoutScroll className={styles.cards}>
        <AnimatePresence initial={false}>
          {column.tasks.map((task) => (
            <BoardCard
              key={task.id}
              task={task}
              isOpen={openID === task.id}
              selected={selectedID === task.id}
              onToggle={() => onToggle(task.id)}
              onOpen={() => onOpen(task.id)}
            />
          ))}
        </AnimatePresence>
        {column.tasks.length === 0 ? (
          <li className={styles.empty}>Nothing here</li>
        ) : null}
      </motion.ul>
    </section>
  );
}
