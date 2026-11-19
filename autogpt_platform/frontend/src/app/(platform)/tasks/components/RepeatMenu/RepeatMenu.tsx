"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  Cancel01Icon,
  MoreHorizontalIcon,
  RepeatOffIcon,
  FlashOffIcon,
} from "@hugeicons/core-free-icons";

interface Props {
  title: string;
  isBusy: boolean;
  /** A task started by an event: it stops watching instead. */
  isWatching?: boolean;
  onStopRepeating: () => void;
  onCancel: () => void;
}

/** The ⋯ beside a repeating or watching task's controls: stop it repeating
 *  or watching (a run going now still finishes), or cancel it altogether. */
export function RepeatMenu({
  title,
  isBusy,
  isWatching = false,
  onStopRepeating,
  onCancel,
}: Props) {
  return (
    <DropdownMenu modal={false}>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost-muted"
          size="icon-xs"
          aria-label={`More for ${title}`}
          withTooltip={false}
          leadingIcon={MoreHorizontalIcon}
          disabled={isBusy}
        />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-44">
        <DropdownMenuItem onSelect={onStopRepeating}>
          <span className="flex items-center gap-2">
            <Icon icon={isWatching ? FlashOffIcon : RepeatOffIcon} size={14} />
            {isWatching ? "Stop watching" : "Stop repeating"}
          </span>
        </DropdownMenuItem>
        <DropdownMenuItem onSelect={onCancel}>
          <span className="flex items-center gap-2">
            <Icon icon={Cancel01Icon} size={14} />
            Cancel task
          </span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
