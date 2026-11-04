import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { SegmentedControl } from "@/components/molecules/SegmentedControl/SegmentedControl";
import {
  DashboardSquare01Icon,
  LeftToRightListBulletIcon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import type { TasksView } from "../../helpers";

interface Props {
  view: TasksView;
  onViewChange: (view: TasksView) => void;
  onNewTask: () => void;
}

const VIEW_OPTIONS: {
  value: TasksView;
  label: string;
  icon: typeof LeftToRightListBulletIcon;
}[] = [
  { value: "list", label: "List", icon: LeftToRightListBulletIcon },
  { value: "board", label: "Board", icon: DashboardSquare01Icon },
];

export function InboxHeader({ view, onViewChange, onNewTask }: Props) {
  return (
    <header className="flex items-center justify-between gap-3">
      <nav
        aria-label="Breadcrumb"
        className="flex min-w-0 items-center gap-1.5"
      >
        <NextLink
          href="/team"
          className="rounded-control transition-opacity duration-150 ease-out-expo hover:opacity-80 focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
        >
          <Text variant="inbox" tone="quiet" as="span">
            Experts
          </Text>
        </NextLink>
        <Text variant="inbox" tone="faint" as="span" aria-hidden>
          /
        </Text>
        <Text variant="inbox-medium" tone="ink" as="h1">
          Tasks
        </Text>
      </nav>
      <div className="flex shrink-0 items-center gap-2">
        <SegmentedControl
          size="sm"
          value={view}
          onChange={onViewChange}
          options={VIEW_OPTIONS}
          ariaLabel="Layout"
        />
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
    </header>
  );
}
