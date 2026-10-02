"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertLlmRouteSection } from "../ExpertLlmRouteSection/ExpertLlmRouteSection";
import { SettingsGroup } from "./components/SettingsGroup";

interface Props {
  expert: Expert;
  onFire: () => void;
}

export function ExpertSettingsSection({ expert, onFire }: Props) {
  return (
    <div className="flex flex-col gap-8">
      <ExpertLlmRouteSection expert={expert} />
      <SettingsGroup
        title="Danger zone"
        label={`${expert.name} danger zone`}
        tone="danger"
      >
        <div className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between sm:gap-6">
          <Text variant="small" tone="danger">
            Firing {expert.name} pauses every schedule and removes them from
            your team.
          </Text>
          <Button
            variant="destructive"
            size="small"
            className="shrink-0 self-start sm:self-auto"
            onClick={onFire}
            data-testid="expert-fire-button"
          >
            Fire {expert.name}
          </Button>
        </div>
      </SettingsGroup>
    </div>
  );
}
