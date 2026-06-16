"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { EyeSlash, TrashSimple } from "@phosphor-icons/react";
import type { CapturedStep } from "../../hooks/useRecordingWorkflow";

interface Props {
  isOpen: boolean;
  steps: CapturedStep[];
  onDeleteStep: (seq: number) => void;
  onRedactStep: (seq: number) => void;
  onApprove: () => void;
  onCancel: () => void;
}

/**
 * Review-before-send view: the captured steps are shown to the user before
 * the recording leaves the machine. The user can delete a step or redact a
 * step's value, then approve. "Demonstration mode requires approval before
 * the recording leaves the machine" (§6) — this is that approval gate.
 *
 * Approval here does NOT necessarily send: for the screenshots_to_cloud
 * route the calling flow then surfaces the §9.1 consent dialog. This view
 * is only about *what* gets sent, not *whether the cloud is allowed*.
 */
export function RecordingReview({
  isOpen,
  steps,
  onDeleteStep,
  onRedactStep,
  onApprove,
  onCancel,
}: Props) {
  return (
    <Dialog
      title="Review what you recorded"
      styling={{ maxWidth: "40rem", minWidth: "auto" }}
      controlled={{
        isOpen,
        set: async (open) => {
          if (!open) onCancel();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-3 py-2">
          <Text variant="body" className="text-sm text-neutral-700">
            These are the steps captured on your machine. Remove anything you
            don&apos;t want included, or hide a value, then approve. Nothing is
            sent until you do.
          </Text>

          {steps.length === 0 ? (
            <Text variant="body" className="text-sm text-neutral-500">
              No steps captured.
            </Text>
          ) : (
            <ul className="flex max-h-80 flex-col gap-1 overflow-y-auto">
              {steps.map((step) => (
                <li
                  key={step.seq}
                  data-testid={`recording-step-${step.seq}`}
                  className="flex items-center justify-between gap-3 rounded-md border border-neutral-200 px-3 py-2"
                >
                  <div className="flex min-w-0 flex-col">
                    <Text
                      variant="body"
                      className="truncate text-sm text-neutral-900"
                    >
                      <span className="font-medium">{step.action}</span>
                      {step.label ? ` · ${step.label}` : ""}
                    </Text>
                    <Text
                      variant="body"
                      className="truncate text-xs text-neutral-500"
                    >
                      {step.activeApp ? `${step.activeApp} · ` : ""}
                      {step.redacted ? (
                        <span className="italic">value hidden</span>
                      ) : step.value ? (
                        <span className="font-mono">{step.value}</span>
                      ) : (
                        ""
                      )}
                    </Text>
                  </div>
                  <div className="flex shrink-0 items-center gap-1">
                    {step.value && !step.redacted ? (
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={`Hide value for step ${step.seq}`}
                        onClick={() => onRedactStep(step.seq)}
                      >
                        <EyeSlash className="h-4 w-4" />
                      </Button>
                    ) : null}
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={`Delete step ${step.seq}`}
                      onClick={() => onDeleteStep(step.seq)}
                    >
                      <TrashSimple className="h-4 w-4 text-red-600" />
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <Dialog.Footer className="justify-end">
          <Button variant="secondary" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={onApprove}
            disabled={steps.length === 0}
          >
            Approve and continue
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
