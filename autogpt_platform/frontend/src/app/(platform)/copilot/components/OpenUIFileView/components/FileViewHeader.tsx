import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DashboardSquare01Icon,
  SidebarRightIcon,
} from "@hugeicons/core-free-icons";

export interface FileViewHeaderProps {
  name: string;
  onOpen: () => void;
}

interface Props extends FileViewHeaderProps {
  status?: string;
}

export function FileViewHeader({ name, onOpen, status }: Props) {
  return (
    <div className="flex min-w-0 items-center gap-2 text-xs text-muted-foreground">
      <Icon icon={DashboardSquare01Icon} size={14} className="shrink-0" />
      <span className="truncate font-medium text-foreground" title={name}>
        {name}
      </span>
      {status && (
        <span role="status" className="shrink-0">
          · {status}
        </span>
      )}
      <Button
        type="button"
        size="xs"
        variant="outline"
        className="ml-auto shrink-0"
        leftIcon={<Icon icon={SidebarRightIcon} size={14} />}
        onClick={onOpen}
      >
        Open
      </Button>
    </div>
  );
}
