"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { ExpertAvatarPicker } from "@/components/molecules/ExpertAvatarPicker/ExpertAvatarPicker";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useExpertAvatarButton } from "./useExpertAvatarButton";

interface Props {
  expert: Expert;
}

export function ExpertAvatarButton({ expert }: Props) {
  const { isOpen, setIsOpen, saveAvatar, isPending } = useExpertAvatarButton(
    expert.id,
  );
  return (
    <>
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        aria-label={`Change ${expert.name}'s photo`}
        className="size-24 shrink-0 rounded-xl focus-visible:ring-2 focus-visible:ring-ring"
      >
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatar_url}
          size={96}
        />
      </button>
      <Dialog
        title={`Change ${expert.name}'s avatar`}
        controlled={{ isOpen, set: setIsOpen }}
      >
        <Dialog.Content>
          {isOpen && (
            <fieldset disabled={isPending} className="min-w-0">
              <ExpertAvatarPicker
                name={expert.name}
                color={expert.color ?? null}
                avatarUrl={expert.avatar_url}
                onPick={saveAvatar}
              />
            </fieldset>
          )}
        </Dialog.Content>
      </Dialog>
    </>
  );
}
