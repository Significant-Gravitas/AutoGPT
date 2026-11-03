import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import type { Task } from "../../data/types";
import { InboxRow } from "../InboxRow/InboxRow";
import { InboxTray } from "./InboxTray";
import { useInboxMotion } from "./useInboxMotion";

interface Props {
  tasks: Task[];
  expandedID: string | null;
  selectedID: string | null;
  onToggle: (id: string) => void;
  onOpen: (id: string) => void;
}

export function InboxList({
  tasks,
  expandedID,
  selectedID,
  onToggle,
  onOpen,
}: Props) {
  const motionProps = useInboxMotion();

  return (
    <InboxTray>
      <LayoutGroup id="inbox-list">
        <ul aria-label="Tasks" className="divide-y divide-border-hairline">
          <AnimatePresence initial={false}>
            {tasks.map((task) => (
              <motion.li key={task.id} {...motionProps}>
                <InboxRow
                  task={task}
                  expanded={expandedID === task.id}
                  selected={selectedID === task.id}
                  onToggle={() => onToggle(task.id)}
                  onOpen={() => onOpen(task.id)}
                  variant="list"
                />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      </LayoutGroup>
    </InboxTray>
  );
}
