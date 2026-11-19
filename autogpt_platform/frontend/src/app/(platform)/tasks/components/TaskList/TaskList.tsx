"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import { Text } from "@/components/atoms/Text/Text";
import { LayoutGroup, motion } from "framer-motion";
import { buildBoardColumns } from "../../helpers";
import { ListGroup } from "./components/ListGroup/ListGroup";

interface Props {
  tasks: Task[];
  experts: Expert[];
  selectedID: string | null;
  onOpen: (id: string) => void;
}

/** The tasks as a list: the board's lanes, in the board's order, stacked
 *  one under the other, one row per task. It scrolls down inside the
 *  height it is given, never sideways. */
export function TaskList({ tasks, experts, selectedID, onOpen }: Props) {
  const groups = buildBoardColumns(tasks).filter(
    (column) => column.tasks.length > 0,
  );

  return (
    <motion.div
      layoutScroll
      className="scrollbar-none min-h-0 flex-1 overflow-x-hidden overflow-y-auto overscroll-y-contain palette-classic"
    >
      <LayoutGroup id="task-list">
        <div className="grid min-w-0 animate-inbox-enter content-start gap-2 pb-1">
          {groups.length === 0 ? (
            <div className="rounded-tray bg-surface-tray py-5 text-center">
              <Text as="span" variant="inbox" tone="faint">
                Nothing here
              </Text>
            </div>
          ) : (
            groups.map((group) => (
              <ListGroup
                key={group.id}
                column={group}
                experts={experts}
                selectedID={selectedID}
                onOpen={onOpen}
              />
            ))
          )}
        </div>
      </LayoutGroup>
    </motion.div>
  );
}
