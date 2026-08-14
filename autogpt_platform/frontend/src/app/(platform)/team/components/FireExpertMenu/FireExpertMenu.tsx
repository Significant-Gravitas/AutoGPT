import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import { Logout03Icon, MoreVerticalIcon } from "@hugeicons/core-free-icons";

interface Props {
  expertName: string;
  onFire: () => void;
  testId: string;
  triggerClassName?: string;
}

export function FireExpertMenu({
  expertName,
  onFire,
  testId,
  triggerClassName,
}: Props) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          aria-label={`${expertName} actions`}
          data-testid={testId}
          onClick={(event) => event.stopPropagation()}
          className={cn(
            "inline-flex h-9 w-9 items-center justify-center rounded-full transition-colors",
            triggerClassName,
          )}
        >
          <Icon icon={MoreVerticalIcon} size={18} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" sideOffset={6}>
        <DropdownMenuItem
          onSelect={onFire}
          className="flex cursor-pointer items-center gap-2 text-red-600 focus:bg-red-50 focus:text-red-700"
        >
          <Icon icon={Logout03Icon} size={14} />
          Fire {expertName}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
