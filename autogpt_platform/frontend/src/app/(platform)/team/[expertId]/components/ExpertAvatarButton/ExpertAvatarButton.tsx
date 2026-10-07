"use client";

import { getExpertTopicHex } from "@/components/molecules/ExpertAvatar/colors";
import { Button } from "@/components/atoms/Button/Button";

import { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import { getExpertVisualCategory } from "@/components/molecules/ExpertAvatar/helpers";
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
      <Button
        type="button"
        variant="ghost"
        unmask={false}
        onClick={() => setIsOpen(true)}
        aria-label={`Change ${expert.name}'s appearance`}
        className="size-24 min-w-0 shrink-0 rounded-full border-0 p-0 hover:border-transparent hover:bg-transparent focus-visible:ring-2 focus-visible:ring-ring"
      >
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatar_url}
          size={96}
          className="rounded-full ring-4 ring-background"
          backgroundColor={getExpertTopicHex({
            avatarUrl: expert.avatar_url,
            categories: expert.categories,
            role: expert.role,
          })}
        />
      </Button>
      <Dialog
        title={`Change ${expert.name}'s avatar`}
        controlled={{ isOpen, set: setIsOpen }}
      >
        <Dialog.Content>
          {isOpen && (
            <fieldset disabled={isPending} className="min-w-0">
              <ExpertAvatarPicker
                name={expert.name}
                category={
                  Object.values(ExpertAvatarRequestCategory).find(
                    (category) =>
                      category ===
                      getExpertVisualCategory(
                        expert.avatar_url,
                        expert.categories,
                      ),
                  ) ?? "general"
                }
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
