"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { EyeSlashIcon, TrashSimpleIcon } from "@phosphor-icons/react";
import type { CapturedStep } from "../../hooks/recording-helpers";

interface MetadataFieldProps {
  label: string;
  value: string;
}

function MetadataField({ label, value }: MetadataFieldProps) {
  return (
    <div className="min-w-0">
      <dt className="text-[11px] font-medium uppercase tracking-wide text-neutral-500">
        {label}
      </dt>
      <dd className="break-all font-mono text-xs text-neutral-800">{value}</dd>
    </div>
  );
}

function capturedValue(value: string | null | undefined) {
  if (value === null || value === undefined) return "Not captured";
  return value.length > 0 ? value : "Empty string";
}

function optionalValue(value: string | number | null | undefined) {
  if (value === null || value === undefined || value === "") {
    return "Not captured";
  }
  return String(value);
}

interface Props {
  isOpen: boolean;
  steps: CapturedStep[];
  isSubmitting?: boolean;
  errorMessage?: string | null;
  onDeleteStep: (seq: number) => void;
  onRedactStep: (seq: number) => void;
  onApprove: () => void;
  onCancel: () => void;
}

/**
 * The native start consent authorizes post-stop transfer of hygiene-redacted
 * structured steps to this authenticated browser. Review applies the user's
 * removals and value redactions to shim-owned data before skill generation.
 * Raw pixels only leave under the separate screenshots-to-cloud consent.
 */
export function RecordingReview({
  isOpen,
  steps,
  isSubmitting = false,
  errorMessage,
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
            Your executor transferred these hygiene-redacted structured steps to
            this authenticated browser for review. Remove anything you
            don&apos;t want included, or hide a value. Finishing applies those
            changes on your executor and unlocks skill generation. Raw
            screenshots stay on your machine unless you separately allow cloud
            processing.
          </Text>

          {errorMessage ? (
            <Text variant="body" role="alert" className="text-sm text-red-700">
              {errorMessage}
            </Text>
          ) : null}

          {steps.length === 0 ? (
            <Text variant="body" className="text-sm text-neutral-500">
              No steps captured.
            </Text>
          ) : (
            <ul className="flex max-h-[26rem] flex-col gap-2 overflow-y-auto pr-1">
              {steps.map((step) => (
                <li
                  key={step.seq}
                  data-testid={`recording-step-${step.seq}`}
                  className="flex flex-col gap-2 rounded-md border border-neutral-200 px-3 py-2 sm:flex-row sm:items-start sm:justify-between"
                >
                  <div className="flex min-w-0 flex-1 flex-col gap-1">
                    <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-neutral-900">
                      <span className="font-medium">
                        Step {step.seq}: {step.action}
                      </span>
                      <span className="rounded-full bg-neutral-100 px-2 py-0.5 text-[11px] font-medium text-neutral-700">
                        outcome: {step.outcome}
                      </span>
                    </div>
                    {step.label ? (
                      <Text
                        variant="body"
                        className="break-words text-sm text-neutral-700"
                      >
                        {step.label}
                      </Text>
                    ) : null}
                    <Text
                      variant="body"
                      className="break-words text-xs text-neutral-500"
                    >
                      {step.activeApp ? `${step.activeApp} · ` : ""}
                      {step.redacted ? (
                        <span className="italic">value hidden</span>
                      ) : step.value !== null && step.value !== undefined ? (
                        <span className="font-mono">
                          {capturedValue(step.value)}
                        </span>
                      ) : (
                        "No value captured"
                      )}
                    </Text>

                    <details className="group mt-1">
                      <summary
                        aria-label={`View retained metadata for step ${step.seq}`}
                        className="w-fit cursor-pointer rounded text-xs font-medium text-neutral-700 outline-none hover:text-neutral-950 focus-visible:ring-2 focus-visible:ring-neutral-500 focus-visible:ring-offset-2"
                      >
                        View retained metadata
                      </summary>
                      <dl className="mt-3 grid grid-cols-1 gap-x-4 gap-y-3 rounded-md bg-neutral-50 p-3 sm:grid-cols-2">
                        <MetadataField
                          label="Sequence"
                          value={String(step.seq)}
                        />
                        <MetadataField
                          label="Timestamp"
                          value={String(step.timestamp)}
                        />
                        <MetadataField label="Actor" value={step.actor} />
                        <MetadataField label="Action" value={step.action} />
                        <MetadataField label="Outcome" value={step.outcome} />
                        <MetadataField
                          label="Redacted"
                          value={step.redacted ? "Yes" : "No"}
                        />
                        <MetadataField
                          label="Screenshot reference"
                          value={optionalValue(step.screenshotRef)}
                        />
                        <MetadataField
                          label="Cursor"
                          value={
                            step.cursor
                              ? `x=${step.cursor[0]}, y=${step.cursor[1]}`
                              : "Not captured"
                          }
                        />
                        <MetadataField
                          label="Active app"
                          value={optionalValue(step.activeApp)}
                        />
                        <MetadataField
                          label="Active window"
                          value={optionalValue(step.activeWindow)}
                        />
                        <MetadataField
                          label="Narration"
                          value={optionalValue(step.narration)}
                        />
                        <MetadataField
                          label="Value"
                          value={
                            step.redacted
                              ? "Hidden by your review"
                              : capturedValue(step.value)
                          }
                        />
                        <MetadataField
                          label="Value type"
                          value={optionalValue(step.valueType)}
                        />
                        <MetadataField
                          label="Parameter"
                          value={
                            step.isParameter === null ||
                            step.isParameter === undefined
                              ? "Not classified"
                              : step.isParameter
                                ? "Yes"
                                : "No"
                          }
                        />
                        <MetadataField
                          label="Enrichment kind"
                          value={step.enrichment.kind}
                        />
                        <MetadataField
                          label="Enrichment label"
                          value={optionalValue(step.enrichment.label)}
                        />
                        <MetadataField
                          label="Enrichment role"
                          value={optionalValue(step.enrichment.role)}
                        />
                        <MetadataField
                          label="Accessibility path"
                          value={optionalValue(step.enrichment.axPath)}
                        />
                        <MetadataField
                          label="Enrichment URL"
                          value={optionalValue(step.enrichment.url)}
                        />
                        <div className="min-w-0 sm:col-span-2">
                          <dt className="text-[11px] font-medium uppercase tracking-wide text-neutral-500">
                            Selectors
                          </dt>
                          <dd className="text-xs text-neutral-800">
                            {step.enrichment.selectors.length > 0 ? (
                              <ul className="mt-1 flex flex-col gap-1">
                                {step.enrichment.selectors.map(
                                  (selector, index) => (
                                    <li
                                      key={`${selector.strategy}-${selector.value}-${index}`}
                                      className="break-all font-mono"
                                    >
                                      {selector.strategy}: {selector.value}
                                    </li>
                                  ),
                                )}
                              </ul>
                            ) : (
                              "Not captured"
                            )}
                          </dd>
                        </div>
                      </dl>
                    </details>
                  </div>
                  <div className="flex shrink-0 items-center justify-end gap-1">
                    {step.value !== null &&
                    step.value !== undefined &&
                    !step.redacted ? (
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={`Hide value for step ${step.seq}`}
                        onClick={() => onRedactStep(step.seq)}
                      >
                        <EyeSlashIcon className="h-4 w-4" aria-hidden="true" />
                      </Button>
                    ) : null}
                    <Button
                      variant="ghost"
                      size="icon"
                      aria-label={`Delete step ${step.seq}`}
                      onClick={() => onDeleteStep(step.seq)}
                    >
                      <TrashSimpleIcon
                        className="h-4 w-4 text-red-600"
                        aria-hidden="true"
                      />
                    </Button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <Dialog.Footer className="flex-wrap justify-end">
          <Button
            variant="secondary"
            disabled={isSubmitting}
            onClick={onCancel}
          >
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={onApprove}
            disabled={steps.length === 0 || isSubmitting}
            loading={isSubmitting}
          >
            Finish review
          </Button>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
