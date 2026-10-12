"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  Delete02Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
} from "@hugeicons/core-free-icons";
import { useBoardRef } from "../../BoardContext";
import { MAX_BOARD_NAME } from "../../board-helpers";
import { useBoardHeader } from "./useBoardHeader";

interface Props {
  name: string | null;
}

export function BoardHeader({ name }: Props) {
  const { isDefault } = useBoardRef();
  const header = useBoardHeader(name ?? "");

  if (name === null) return null;

  if (header.isRenaming) {
    return (
      <Input
        id="board-name"
        label="Project name"
        hideLabel
        size="small"
        autoFocus
        autoComplete="off"
        maxLength={MAX_BOARD_NAME}
        value={header.draft}
        error={header.error ?? undefined}
        onChange={header.handleChange}
        onKeyDown={header.handleKeyDown}
        onBlur={header.cancelRename}
        wrapperClassName="mb-0 w-72 max-w-full"
      />
    );
  }

  return (
    <div className="flex min-w-0 items-center gap-1">
      <Text variant="h4" as="h1" className="truncate">
        {name}
      </Text>
      {!isDefault && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon-xs"
              aria-label="Project actions"
              className="text-zinc-500 hover:bg-zinc-200/70 hover:text-zinc-800"
            >
              <Icon icon={MoreHorizontalIcon} size={16} />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="start"
            className="w-44"
            onCloseAutoFocus={header.handleMenuCloseAutoFocus}
          >
            <DropdownMenuItem onSelect={header.chooseRename}>
              <Icon icon={PencilEdit02Icon} size={16} />
              Rename
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onSelect={header.chooseDelete}
              className="text-red-600 focus:text-red-700"
            >
              <Icon icon={Delete02Icon} size={16} />
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
      {header.isDeleteOpen && (
        <Dialog
          title={`Delete “${name}”?`}
          styling={{ maxWidth: "28rem" }}
          controlled={{
            isOpen: true,
            set: (open) => {
              if (!open) header.cancelDelete();
            },
          }}
        >
          <Dialog.Content>
            <div className="flex flex-col gap-4 px-1">
              <Text variant="body" className="text-zinc-700">
                Its tasks move to Home&apos;s Backlog.
              </Text>
              <div className="flex justify-end gap-2 pt-2">
                <Button
                  variant="secondary"
                  onClick={header.cancelDelete}
                  disabled={header.isDeleting}
                >
                  Cancel
                </Button>
                <Button
                  variant="destructive"
                  onClick={header.confirmDelete}
                  loading={header.isDeleting}
                >
                  Delete
                </Button>
              </div>
            </div>
          </Dialog.Content>
        </Dialog>
      )}
    </div>
  );
}
