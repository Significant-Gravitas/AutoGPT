import { Clock01Icon, ComputerIcon, Tick02Icon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/Icon";
import { PanelEmptyState } from "@/components/PanelEmptyState";
import { useColors } from "@/lib/theme";

interface Props {
  expertName: string;
}

// One tile per thing the tab lists: upcoming routines, finished runs, and
// the expert's computer.
export function ActivityEmptyState({ expertName }: Props) {
  const colors = useColors();
  return (
    <PanelEmptyState
      tiles={[
        <Icon key="upcoming" icon={Clock01Icon} size={22} color={colors.sky} />,
        <Icon key="computer" icon={ComputerIcon} size={22} color={colors.success} />,
        <Icon key="recent" icon={Tick02Icon} size={22} color={colors.brand} />,
      ]}
      title="Nothing to show yet"
      description={`When ${expertName} runs a routine or finishes a task, it shows up here.`}
    />
  );
}
