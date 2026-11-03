import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  TabsLine,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import {
  DashboardSquare01Icon,
  LeftToRightListBulletIcon,
  PlusSignIcon,
  UserIcon,
  ViewIcon,
  ViewOffSlashIcon,
} from "@hugeicons/core-free-icons";
import type { ExpertOption } from "../../data/types";
import { TASK_VIEWS, type PageFilters, type TasksView } from "../../helpers";
import { FilterMenu } from "../FilterMenu/FilterMenu";

interface Props {
  view: TasksView;
  onViewChange: (view: TasksView) => void;
  filters: PageFilters;
  onFiltersChange: (filters: PageFilters) => void;
  experts: ExpertOption[];
  showHidden: boolean;
  onShowHiddenChange: (show: boolean) => void;
  onNewTask: () => void;
}

function isView(value: string): value is TasksView {
  return (TASK_VIEWS as readonly string[]).includes(value);
}

export function TasksHeader({
  view,
  onViewChange,
  filters,
  onFiltersChange,
  experts,
  showHidden,
  onShowHiddenChange,
  onNewTask,
}: Props) {
  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between gap-4">
        <Text variant="h3" as="h1">
          Tasks
        </Text>
        <Button
          variant="primary"
          size="small"
          leftIcon={<Icon icon={PlusSignIcon} size={16} />}
          onClick={onNewTask}
          aria-keyshortcuts="N"
        >
          New task
        </Button>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <TabsLine
          variant="pill"
          value={view}
          onValueChange={(value) => {
            if (isView(value)) onViewChange(value);
          }}
        >
          <TabsLineList>
            <TabsLineTrigger value="board" icon={DashboardSquare01Icon}>
              Board
            </TabsLineTrigger>
            <TabsLineTrigger value="list" icon={LeftToRightListBulletIcon}>
              List
            </TabsLineTrigger>
            <TabsLineTrigger value="mine" icon={UserIcon}>
              My tasks
            </TabsLineTrigger>
          </TabsLineList>
        </TabsLine>
        <div className="flex items-center gap-2">
          <FilterMenu
            filters={filters}
            onChange={onFiltersChange}
            experts={experts}
          />
          <Button
            variant="toggle"
            size="xs"
            aria-pressed={showHidden}
            leftIcon={
              <Icon icon={showHidden ? ViewIcon : ViewOffSlashIcon} size={14} />
            }
            onClick={() => onShowHiddenChange(!showHidden)}
          >
            Hidden
          </Button>
        </div>
      </div>
    </div>
  );
}
