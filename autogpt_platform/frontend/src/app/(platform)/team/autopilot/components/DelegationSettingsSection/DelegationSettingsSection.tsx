"use client";

import type { DelegationSettingsUpdateMode } from "@/app/api/__generated__/models/delegationSettingsUpdateMode";
import { Select } from "@/components/atoms/Select/Select";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Switch } from "@/components/atoms/Switch/Switch";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import {
  getBudgetOptions,
  getCapOptions,
  MODE_OPTIONS,
  TOGGLES,
} from "./helpers";
import { SettingRow } from "./SettingRow";
import { useDelegationSettings } from "./useDelegationSettings";

interface Props {
  enabled: boolean;
}

const SELECT_CLASS = "h-9 w-full rounded-lg py-1.5 pl-3 pr-2.5 sm:w-40";

export function DelegationSettingsSection({ enabled }: Props) {
  const { settings, isLoading, isError, refetch, update } =
    useDelegationSettings({ enabled });

  if (isLoading) {
    return (
      <div className="max-w-[900px] space-y-3 pt-4">
        <Skeleton className="h-20 w-full rounded-lg" />
        <Skeleton className="h-20 w-full rounded-lg" />
        <Skeleton className="h-20 w-full rounded-lg" />
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorCard
        context="delegation settings"
        hint="We could not load how Otto delegates."
        onRetry={() => refetch()}
      />
    );
  }

  return (
    <section aria-label="Delegation settings" className="max-w-[900px] pt-2">
      <SettingRow
        id="delegation-mode"
        title="Delegation mode"
        description="How Otto hands work to your experts."
        helper="Ask first proposes each hand-off for approval · Auto lets a judge check each call · Unsupervised runs within caps"
      >
        <Select
          id="delegation-mode-select"
          label="Delegation mode"
          hideLabel
          size="small"
          className={SELECT_CLASS}
          value={settings.mode}
          options={MODE_OPTIONS}
          onValueChange={(mode) =>
            void update({ mode: mode as DelegationSettingsUpdateMode })
          }
        />
      </SettingRow>
      <SettingRow
        id="delegation-cap"
        title="Per-delegation cap"
        description="The most Otto can spend on a single hand-off."
        helper="Weekly caps live on each expert"
      >
        <Select
          id="delegation-cap-select"
          label="Per-delegation cap"
          hideLabel
          size="small"
          className={SELECT_CLASS}
          value={String(settings.per_delegation_cap_usd)}
          options={getCapOptions(settings.per_delegation_cap_usd)}
          onValueChange={(value) =>
            void update({ per_delegation_cap_usd: Number(value) })
          }
        />
      </SettingRow>
      <SettingRow
        id="delegation-budget"
        title="Daily delegation budget"
        description="Otto stops delegating for the day once this is used up."
      >
        <Select
          id="delegation-budget-select"
          label="Daily delegation budget"
          hideLabel
          size="small"
          className={SELECT_CLASS}
          value={String(settings.daily_budget_usd)}
          options={getBudgetOptions(settings.daily_budget_usd)}
          onValueChange={(value) =>
            void update({ daily_budget_usd: Number(value) })
          }
        />
      </SettingRow>
      {TOGGLES.map((toggle) => (
        <SettingRow
          key={toggle.key}
          id={toggle.key}
          title={toggle.title}
          description={toggle.description}
        >
          <Switch
            checked={settings[toggle.key]}
            aria-labelledby={`${toggle.key}-title`}
            aria-describedby={`${toggle.key}-description`}
            onCheckedChange={(checked) =>
              void update({ [toggle.key]: checked })
            }
          />
        </SettingRow>
      ))}
    </section>
  );
}
