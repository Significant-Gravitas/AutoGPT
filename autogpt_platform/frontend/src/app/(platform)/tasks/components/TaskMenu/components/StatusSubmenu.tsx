import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { statusOf } from "../../../helpers";

interface Props {
  value: TaskStatus;
  statuses: TaskStatus[];
  onPick: (value: string) => void;
}

/** The statuses a person can set by hand. */
export function StatusSubmenu({ value, statuses, onPick }: Props) {
  return (
    <DropdownMenuSub>
      <DropdownMenuSubTrigger>
        <span className="flex items-center gap-2">
          <Icon icon={Tick02Icon} size={14} />
          Change status
        </span>
      </DropdownMenuSubTrigger>
      <DropdownMenuSubContent className="w-40">
        <DropdownMenuRadioGroup value={value} onValueChange={onPick}>
          {statuses.map((status) => (
            <DropdownMenuRadioItem key={status} value={status}>
              {statusOf(status).word}
            </DropdownMenuRadioItem>
          ))}
        </DropdownMenuRadioGroup>
      </DropdownMenuSubContent>
    </DropdownMenuSub>
  );
}
