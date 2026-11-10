import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import { ownerOf } from "../../helpers";
import { TaskRow } from "../TaskRow/TaskRow";
import { TasksCard } from "./TasksCard";
import { useListMotion } from "./useListMotion";

interface Props {
  tasks: Task[];
  experts: Expert[];
  selectedID: string | null;
  onOpen: (id: string) => void;
}

export function TaskList({ tasks, experts, selectedID, onOpen }: Props) {
  const motionProps = useListMotion();

  return (
    <TasksCard>
      <LayoutGroup id="task-list">
        <ul aria-label="Tasks" className="divide-y divide-border-hairline">
          <AnimatePresence initial={false} mode="popLayout">
            {tasks.map((task) => (
              <motion.li key={task.id} {...motionProps}>
                <TaskRow
                  task={task}
                  owner={ownerOf(task, experts)}
                  selected={selectedID === task.id}
                  onOpen={() => onOpen(task.id)}
                />
              </motion.li>
            ))}
          </AnimatePresence>
        </ul>
      </LayoutGroup>
    </TasksCard>
  );
}
