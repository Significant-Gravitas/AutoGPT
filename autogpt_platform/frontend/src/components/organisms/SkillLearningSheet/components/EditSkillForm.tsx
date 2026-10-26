"use client";

import { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { lineDiff } from "@/services/skill-learning/helpers";
import { SkillEditDraft } from "../useSkillReviewState";

interface Props {
  currentVersion: SkillVersionSummary | null;
  draft: SkillEditDraft;
  isBusy: boolean;
  onDraftChange: (patch: Partial<SkillEditDraft>) => void;
  onReviewCurrent: () => void;
  onSave: () => void;
  onCancel: () => void;
}

export function EditSkillForm({
  currentVersion,
  draft,
  isBusy,
  onDraftChange,
  onReviewCurrent,
  onSave,
  onCancel,
}: Props) {
  const hasNewerVersion = draft.baseVersionId !== (currentVersion?.id ?? null);
  return (
    <form
      data-slot="edit-skill-form"
      className="flex flex-col gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        if (!hasNewerVersion && !isBusy) onSave();
      }}
      aria-label="Edit skill"
    >
      <Text variant="small" tone="muted">
        Your edit takes effect immediately and becomes a new version. The same
        content checks apply.
      </Text>
      {hasNewerVersion && currentVersion ? (
        <section
          aria-label="Newer version available"
          data-slot="edit-skill-form-conflict"
          className="rounded-lg border border-warning-border bg-warning-surface p-3"
        >
          <Text variant="small" tone="primary">
            v{currentVersion.version} was saved while you were editing. Your
            draft is kept. Compare it with the current version before saving.
          </Text>
          <details
            data-slot="edit-skill-form-conflict-details"
            className="mt-2"
          >
            <summary
              data-slot="edit-skill-form-conflict-summary"
              className="cursor-pointer text-sm font-medium"
            >
              Review changes against v{currentVersion.version}
            </summary>
            {draft.description !== currentVersion.description ? (
              <Text variant="small" tone="secondary" className="mt-2">
                Current description: {currentVersion.description}
              </Text>
            ) : null}
            <pre
              data-slot="edit-skill-form-conflict-diff"
              // TODO(tokens): no foreground role for zinc-100 (#EFEFF0) on the
              // dark code surface.
              className="my-3 overflow-x-auto rounded-lg bg-action-hover p-3 text-xs text-zinc-100"
            >
              {lineDiff(currentVersion.body ?? "", draft.body) ||
                "Your procedure matches the current version."}
            </pre>
            <Button
              type="button"
              variant="secondary"
              size="small"
              disabled={isBusy}
              onClick={onReviewCurrent}
            >
              Continue editing from v{currentVersion.version}
            </Button>
          </details>
        </section>
      ) : null}
      <Input
        id="skill-edit-description"
        label="Description"
        labelVariant="small-medium"
        value={draft.description}
        maxLength={1024}
        required
        disabled={isBusy}
        onChange={(event) => onDraftChange({ description: event.target.value })}
      />
      <Input
        id="skill-edit-body"
        label="Procedure (Markdown)"
        labelVariant="small-medium"
        type="textarea"
        rows={12}
        value={draft.body}
        required
        disabled={isBusy}
        onChange={(event) => onDraftChange({ body: event.target.value })}
      />
      <label
        data-slot="edit-skill-form-auto-improve"
        className="flex items-center justify-between gap-3"
      >
        <span
          data-slot="edit-skill-form-auto-improve-text"
          className="flex flex-col"
        >
          <Text variant="small" tone="primary">
            Keep automatic improvements on
          </Text>
          <Text variant="small" tone="muted">
            Off by default after an edit: later automated changes become
            proposals for you to decide.
          </Text>
        </span>
        <Switch
          checked={draft.keepAutoImprove}
          disabled={isBusy}
          aria-label="Keep automatic improvements on"
          onCheckedChange={(checked) =>
            onDraftChange({ keepAutoImprove: checked })
          }
        />
      </label>
      <div
        data-slot="edit-skill-form-actions"
        className="flex justify-end gap-2"
      >
        <Button
          type="button"
          variant="ghost"
          size="small"
          disabled={isBusy}
          onClick={onCancel}
        >
          Cancel
        </Button>
        <Button
          type="submit"
          variant="primary"
          size="small"
          loading={isBusy}
          disabled={hasNewerVersion}
        >
          Save edit
        </Button>
      </div>
    </form>
  );
}
