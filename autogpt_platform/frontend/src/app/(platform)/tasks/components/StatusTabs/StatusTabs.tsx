import type { Task } from "@/app/api/__generated__/models/task";
import { SegmentedControl } from "@/components/molecules/SegmentedControl/SegmentedControl";
import { TAB_LABELS, TASK_TABS, tabCount, type TaskTab } from "../../helpers";

interface Props {
  tab: TaskTab;
  onTabChange: (tab: TaskTab) => void;
  tasks: Task[];
}

export function StatusTabs({ tab, onTabChange, tasks }: Props) {
  const options = TASK_TABS.map((value) => ({
    value,
    label: TAB_LABELS[value],
    count: tabCount(tasks, value),
  }));

  return (
    <div className="-mx-1 scrollbar-none max-w-full overflow-x-auto px-1">
      <SegmentedControl
        size="sm"
        value={tab}
        onChange={onTabChange}
        options={options}
        ariaLabel="Show tasks"
      />
    </div>
  );
}
