"use client";

import type { ScheduledChatCadence } from "@/app/api/__generated__/models/scheduledChatCadence";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { cn } from "@/lib/utils";
import { useId } from "react";
import { lowerFirst } from "./helpers";
import { useRunLessOftenDialog } from "./useRunLessOftenDialog";

interface Props {
  scheduleKey: string;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

export function RunLessOftenDialog({ scheduleKey, isOpen, setIsOpen }: Props) {
  const dialog = useRunLessOftenDialog({
    scheduleKey,
    isOpen,
    onDone: () => setIsOpen(false),
  });
  const canChoose = (dialog.cadence?.options.length ?? 0) > 0;

  return (
    <Dialog
      controlled={{ isOpen, set: setIsOpen }}
      styling={{ maxWidth: "28rem" }}
      title={
        dialog.cadence
          ? `Run “${dialog.cadence.title}” less often`
          : "Run less often"
      }
    >
      <Dialog.Content>
        {dialog.isLoading ? (
          <div className="flex flex-col gap-2" data-testid="cadence-loading">
            <Skeleton className="h-10 w-full" />
            <Skeleton className="h-10 w-full" />
          </div>
        ) : dialog.refusal ? (
          <Text variant="body" className="text-foreground-secondary">
            {dialog.refusal}
          </Text>
        ) : dialog.isError || !dialog.cadence ? (
          <ErrorCard
            responseError={{ message: "Couldn't load how often it runs." }}
            context="its schedule"
          />
        ) : (
          <CadenceChoices
            cadence={dialog.cadence}
            selected={dialog.selected}
            onSelect={dialog.select}
          />
        )}
        <Dialog.Footer>
          <Button
            type="button"
            variant="secondary"
            size="small"
            onClick={() => setIsOpen(false)}
          >
            Cancel
          </Button>
          {canChoose ? (
            <Button
              type="button"
              variant="primary"
              size="small"
              disabled={!dialog.selected}
              loading={dialog.isSaving}
              onClick={dialog.save}
            >
              Save
            </Button>
          ) : null}
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}

interface ChoicesProps {
  cadence: ScheduledChatCadence;
  selected: string | null;
  onSelect: (cron: string) => void;
}

function CadenceChoices({ cadence, selected, onSelect }: ChoicesProps) {
  const groupName = useId();

  if (cadence.options.length === 0) {
    return (
      <Text variant="body" className="text-foreground-secondary">
        “{cadence.title}” already runs once a week or less, so there is nothing
        less often to switch it to.
      </Text>
    );
  }
  return (
    <div className="flex flex-col gap-3">
      <Text variant="body" className="text-foreground-secondary">
        {cadence.current ? `It runs ${lowerFirst(cadence.current)} now. ` : ""}
        Fewer runs use less of your allowance. Times are in {cadence.timezone}.
      </Text>
      <div
        role="radiogroup"
        aria-label="How often it runs"
        className="flex flex-col gap-2"
      >
        {cadence.options.map((option) => (
          <label
            key={option.cron}
            className={cn(
              "flex cursor-pointer items-center gap-3 rounded-lg border px-3 py-2.5 text-sm text-foreground-strong",
              selected === option.cron
                ? "border-action-hover bg-surface-muted"
                : "border-border-base hover:bg-surface-muted",
            )}
          >
            <input
              type="radio"
              name={groupName}
              value={option.cron}
              checked={selected === option.cron}
              onChange={() => onSelect(option.cron)}
              className="size-4 accent-action-hover"
            />
            {option.label}
          </label>
        ))}
      </div>
    </div>
  );
}
