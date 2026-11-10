"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import { LayoutGroup } from "framer-motion";
import { buildBoardColumns } from "../../helpers";
import { BoardLane } from "./components/BoardLane/BoardLane";
import { useOverflowRight } from "./useOverflowRight";

interface Props {
  tasks: Task[];
  experts: Expert[];
  selectedID: string | null;
  onOpen: (id: string) => void;
}

/** The tasks board: four lanes side by side that scroll sideways, with
 *  cards that glide to their new lane when their status changes. */
export function TaskBoard({ tasks, experts, selectedID, onOpen }: Props) {
  const { ref: scrollerRef, hiddenRight } = useOverflowRight<HTMLDivElement>();
  const columns = buildBoardColumns(tasks);

  return (
    <div className="relative palette-classic">
      <div
        ref={scrollerRef}
        className="scrollbar-none overflow-x-auto overscroll-x-contain pb-1"
      >
        <LayoutGroup id="task-board">
          <div className="flex w-max min-w-full animate-inbox-enter items-start gap-2">
            {columns.map((column) => (
              <BoardLane
                key={column.id}
                column={column}
                experts={experts}
                selectedID={selectedID}
                onOpen={onOpen}
              />
            ))}
          </div>
        </LayoutGroup>
      </div>
      {hiddenRight ? (
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 right-0 w-16 edge-blur-right"
        />
      ) : null}
    </div>
  );
}
