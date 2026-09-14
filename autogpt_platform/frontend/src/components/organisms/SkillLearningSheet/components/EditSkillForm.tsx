"use client";

import { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";

interface Props {
  version: SkillVersionSummary;
  draft: string;
  isBusy: boolean;
  onDraftChange: (value: string) => void;
  onSave: (body: string, description: string, keepAutoImprove: boolean) => void;
  onCancel: () => void;
}

export function EditSkillForm({
  version,
  draft,
  isBusy,
  onDraftChange,
  onSave,
  onCancel,
}: Props) {
  const [description, setDescription] = useState(version.description);
  const [keepAutoImprove, setKeepAutoImprove] = useState(false);
  const body = draft;
  return (
    <form
      className="flex flex-col gap-3"
      onSubmit={(event) => {
        event.preventDefault();
        onSave(body, description, keepAutoImprove);
      }}
      aria-label="Edit skill"
    >
      <Text variant="small" tone="muted">
        Your edit takes effect immediately and becomes a new version. The same
        content checks apply.
      </Text>
      <Input
        id="skill-edit-description"
        label="Description"
        labelVariant="small-medium"
        value={description}
        maxLength={1024}
        required
        onChange={(event) => setDescription(event.target.value)}
      />
      <Input
        id="skill-edit-body"
        label="Procedure (Markdown)"
        labelVariant="small-medium"
        type="textarea"
        rows={12}
        value={body}
        required
        onChange={(event) => onDraftChange(event.target.value)}
      />
      <label className="flex items-center justify-between gap-3">
        <span className="flex flex-col">
          <Text variant="small" tone="primary">
            Keep automatic improvements on
          </Text>
          <Text variant="small" tone="muted">
            Off by default after an edit: later automated changes become
            proposals for you to decide.
          </Text>
        </span>
        <Switch
          checked={keepAutoImprove}
          aria-label="Keep automatic improvements on"
          onCheckedChange={setKeepAutoImprove}
        />
      </label>
      <div className="flex justify-end gap-2">
        <Button type="button" variant="ghost" size="small" onClick={onCancel}>
          Cancel
        </Button>
        <Button type="submit" variant="primary" size="small" loading={isBusy}>
          Save edit
        </Button>
      </div>
    </form>
  );
}
