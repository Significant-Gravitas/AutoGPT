"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { PublishExpertRow } from "./PublishExpertRow/PublishExpertRow";

interface Props {
  expert: Expert;
  canPublish: boolean;
  onFire: () => void;
}

export function ExpertSettingsSection({ expert, canPublish, onFire }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <PublishExpertRow expert={expert} enabled={canPublish} />
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
    </div>
  );
}
