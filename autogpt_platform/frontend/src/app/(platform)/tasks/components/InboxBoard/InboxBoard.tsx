"use client";

import { LayoutGroup } from "framer-motion";
import { useState } from "react";
import type { Task } from "../../data/types";
import { buildBoardColumns } from "../../helpers";
import styles from "./board.module.css";
import { BoardLane } from "./components/BoardLane";
import { useOverflowRight } from "./useOverflowRight";

interface Props {
  tasks: Task[];
  selectedID: string | null;
  onOpen: (id: string) => void;
}

/** The tasks board: four lanes side by side that scroll sideways, cards
 *  that unfold in place and glide to their new lane when their state changes. */
export function InboxBoard({ tasks, selectedID, onOpen }: Props) {
  const [openID, setOpenID] = useState<string | null>(null);
  const scroller = useOverflowRight<HTMLDivElement>();
  const columns = buildBoardColumns(tasks);

  function toggle(id: string) {
    setOpenID((current) => (current === id ? null : id));
  }

  return (
    <div className="relative h-full min-h-0">
      <div
        ref={scroller.ref}
        className="scrollbar-none h-full min-h-0 overflow-auto overscroll-x-contain"
      >
        <LayoutGroup id="inbox-board">
          <div className={styles.lanes}>
            {columns.map((column) => (
              <BoardLane
                key={column.id}
                column={column}
                openID={openID}
                selectedID={selectedID}
                onToggle={toggle}
                onOpen={onOpen}
              />
            ))}
          </div>
        </LayoutGroup>
      </div>
      {scroller.hiddenRight ? (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 right-0 w-16 edge-blur-right"
        />
      ) : null}
    </div>
  );
}
