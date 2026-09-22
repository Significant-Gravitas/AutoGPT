"use client";

import { Button } from "@/components/atoms/Button/Button";

interface Props {
  visible: boolean;
  saving: boolean;
  onDiscard: () => void;
  onSave: () => void;
}

export function SaveBar({ visible, saving, onDiscard, onSave }: Props) {
  return (
    <div className="flex items-center justify-end gap-2 px-4">
      <Button
        variant="primary"
        size="large"
        onClick={onSave}
        loading={saving}
        disabled={saving || !visible}
      >
        Save changes
      </Button>
      <Button
        variant="secondary"
        size="large"
        onClick={onDiscard}
        disabled={saving || !visible}
      >
        Discard
      </Button>
    </div>
  );
}
