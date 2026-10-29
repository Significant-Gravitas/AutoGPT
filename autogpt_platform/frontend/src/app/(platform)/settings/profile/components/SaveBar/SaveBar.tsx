"use client";

import { Button } from "@/components/atoms/Button/Button";

interface Props {
  dirty: boolean;
  saving: boolean;
  canSave: boolean;
  onDiscard: () => void;
  onSave: () => void;
}

export function SaveBar({ dirty, saving, canSave, onDiscard, onSave }: Props) {
  return (
    <div className="flex items-center justify-start gap-2">
      <Button
        variant="outline"
        size="large"
        onClick={onDiscard}
        disabled={!dirty || saving}
      >
        Discard
      </Button>
      <Button
        variant="primary"
        size="large"
        onClick={onSave}
        loading={saving}
        disabled={!canSave || saving}
      >
        Save changes
      </Button>
    </div>
  );
}
