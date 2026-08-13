"use client";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { LeftToRightListBulletIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  children: React.ReactNode;
}

export function StepsCollapse({ children }: Props) {
  return (
    <Dialog title="Steps">
      <Dialog.Trigger>
        <button
          type="button"
          className="flex items-center gap-1 !text-xs text-zinc-500 transition-colors hover:text-zinc-700"
        >
          <Icon icon={LeftToRightListBulletIcon} size={12} />
          <span>Show steps</span>
        </button>
      </Dialog.Trigger>
      <Dialog.Content>
        <div className="space-y-1">{children}</div>
      </Dialog.Content>
    </Dialog>
  );
}
