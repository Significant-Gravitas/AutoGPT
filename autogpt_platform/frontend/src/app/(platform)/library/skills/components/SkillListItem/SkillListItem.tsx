"use client";

import type { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { BookOpenIcon, TrashIcon } from "@phosphor-icons/react";
import { useSkillListItem } from "./useSkillListItem";

interface Props {
  skill: CopilotSkillInfo;
}

export function SkillListItem({ skill }: Props) {
  const {
    descriptionPreview,
    triggers,
    isDeleteOpen,
    openDelete,
    closeDelete,
    isDeleting,
    handleDelete,
  } = useSkillListItem({ skill });

  return (
    <div
      className="flex w-full flex-col gap-3 rounded-large border border-zinc-200 bg-white p-4 sm:flex-row sm:items-start sm:justify-between"
      data-testid="skill-row"
      data-skill-name={skill.name}
    >
      <div className="flex min-w-0 flex-1 items-start gap-3">
        <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-large border border-slate-50 bg-violet-50">
          <BookOpenIcon size={18} className="text-violet-700" weight="bold" />
        </div>
        <div className="flex min-w-0 flex-col gap-1">
          <Text variant="body-medium" className="break-words">
            {skill.name}
          </Text>
          <Text variant="small" className="!text-zinc-500">
            {descriptionPreview}
          </Text>
          {triggers.length > 0 ? (
            <div
              className="mt-1 flex flex-wrap gap-1"
              data-testid="skill-triggers"
            >
              {triggers.map((trigger) => (
                <span
                  key={trigger}
                  className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-600"
                >
                  {trigger}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      </div>

      <div className="flex flex-shrink-0 items-center gap-2">
        <Button
          variant="secondary"
          size="small"
          onClick={openDelete}
          data-testid="skill-delete-button"
          aria-label="Delete skill"
        >
          <TrashIcon className="mr-1 h-4 w-4" />
          Delete
        </Button>
      </div>

      <Dialog
        controlled={{ isOpen: isDeleteOpen, set: closeDelete }}
        styling={{ maxWidth: "32rem" }}
        title="Delete skill"
      >
        <Dialog.Content>
          <div className="flex flex-col gap-4">
            <Text variant="large">
              Delete the skill <strong>{skill.name}</strong>? Your copilot will
              forget this procedure and can re-distill it later if needed.
            </Text>
            <Dialog.Footer>
              <Button
                variant="secondary"
                disabled={isDeleting}
                onClick={() => closeDelete(false)}
              >
                Keep it
              </Button>
              <Button
                variant="destructive"
                onClick={handleDelete}
                loading={isDeleting}
                data-testid="skill-confirm-delete"
              >
                Yes, delete
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}
