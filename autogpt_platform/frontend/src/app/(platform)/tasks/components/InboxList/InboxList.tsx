import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import type { Task } from "../../data/types";
import { InboxRow } from "../InboxRow/InboxRow";
import { InboxCard } from "./InboxCard";
import { useInboxMotion } from "./useInboxMotion";

interface Props {
  tasks: Task[];
  selectedID: string | null;
  onOpen: (id: string) => void;
}

export function InboxList({ tasks, selectedID, onOpen }: Props) {
  const motionProps = useInboxMotion();

  return (
    <InboxCard>
      <LayoutGroup id="inbox-list">
        <ul aria-label="Tasks" className="divide-y divide-border-hairline">
          <AnimatePresence initial={false}>
            {tasks.map((task) => (
              <motion.li key={task.id} {...motionProps}>
                <InboxRow
                  task={task}
                  selected={selectedID === task.id}
                  onOpen={() => onOpen(task.id)}
                  variant="list"
                />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      </LayoutGroup>
    </InboxCard>
  );
}
