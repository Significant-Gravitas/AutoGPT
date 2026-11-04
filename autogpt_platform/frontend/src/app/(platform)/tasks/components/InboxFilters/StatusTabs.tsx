import { SegmentedControl } from "@/components/molecules/SegmentedControl/SegmentedControl";
import type { TaskCounts } from "../../data/types";
import {
  INBOX_TAB_LABELS,
  INBOX_TABS,
  tabCount,
  type InboxTab,
} from "../../helpers";

interface Props {
  tab: InboxTab;
  onTabChange: (tab: InboxTab) => void;
  counts: TaskCounts | undefined;
}

export function StatusTabs({ tab, onTabChange, counts }: Props) {
  const options = INBOX_TABS.map((value) => ({
    value,
    label: INBOX_TAB_LABELS[value],
    count: tabCount(counts, value),
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
