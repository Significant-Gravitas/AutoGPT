"use client";

import type { ResolvedPage } from "@/app/api/__generated__/models/resolvedPage";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { isKey } from "@/lib/keyboard";
import {
  ArrowTurnBackwardIcon,
  Delete02Icon,
  MoreHorizontalIcon,
  PencilEdit02Icon,
  PinOffIcon,
} from "@hugeicons/core-free-icons";
import { useState } from "react";

interface Props {
  page: ResolvedPage;
  onRename(title: string): void;
  onRestore(): void;
  onUnpin(): void;
  onDelete(): void;
}

export function PageMenu({
  page,
  onRename,
  onRestore,
  onUnpin,
  onDelete,
}: Props) {
  const [isRenaming, setIsRenaming] = useState(false);
  const [title, setTitle] = useState(page.title);

  function submit() {
    const next = title.trim();
    if (!next) return;
    onRename(next);
    setIsRenaming(false);
  }

  return (
    <>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button variant="ghost" size="icon-xs" aria-label="Page actions">
            <Icon icon={MoreHorizontalIcon} size={18} />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end" className="w-48">
          <DropdownMenuItem
            onSelect={() => {
              setTitle(page.title);
              setIsRenaming(true);
            }}
          >
            <Icon icon={PencilEdit02Icon} size={16} className="mr-2" />
            Rename
          </DropdownMenuItem>
          {page.has_previous ? (
            <DropdownMenuItem onSelect={onRestore}>
              <Icon icon={ArrowTurnBackwardIcon} size={16} className="mr-2" />
              Undo last change
            </DropdownMenuItem>
          ) : null}
          {page.pinned_library_agent_id ? (
            <DropdownMenuItem onSelect={onUnpin}>
              <Icon icon={PinOffIcon} size={16} className="mr-2" />
              Unpin from its workflow
            </DropdownMenuItem>
          ) : null}
          <DropdownMenuSeparator />
          <DropdownMenuItem onSelect={onDelete}>
            <Icon icon={Delete02Icon} size={16} className="mr-2" />
            Delete page
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <Dialog
        controlled={{ isOpen: isRenaming, set: setIsRenaming }}
        styling={{ maxWidth: "28rem" }}
        title="Rename page"
      >
        <Dialog.Content>
          <div className="flex flex-col gap-3">
            <Input
              id="rename-page-title"
              label="Title"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              onKeyDown={(e) => {
                if (isKey(e, "Enter")) submit();
              }}
              className="w-full"
              wrapperClassName="!mb-0"
            />
            <Button variant="primary" disabled={!title.trim()} onClick={submit}>
              Save
            </Button>
          </div>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
