"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  expert: Expert;
  onFire: () => void;
}

export function ExpertSettingsSection({ expert, onFire }: Props) {
  return (
    <section className="rounded-xl border border-red-200 bg-red-50/50 p-4">
      <Text variant="large-medium" tone="danger">
        Danger zone
      </Text>
      <Text variant="small" tone="danger" className="mt-1">
        Firing {expert.name} pauses every schedule and removes them from your
        team.
      </Text>
      <Button
        variant="destructive"
        size="small"
        className="mt-4"
        onClick={onFire}
        data-testid="expert-fire-button"
      >
        Fire {expert.name}
      </Button>
    </section>
  );
}
