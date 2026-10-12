"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Link } from "@/components/atoms/Link/Link";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { SettingsGroup } from "../ExpertSettingsSection/components/SettingsGroup";
import { SettingsRow } from "../ExpertSettingsSection/components/SettingsRow";
import { useExpertLlmRouteSection } from "./useExpertLlmRouteSection";

interface Props {
  expert: Expert;
}

export function ExpertLlmRouteSection({ expert }: Props) {
  const { value, options, note, isLoading, isSaving, selectRoute } =
    useExpertLlmRouteSection({ expert });
  const selectId = `expert-${expert.id}-llm-route`;
  const noteId = `${selectId}-note`;

  return (
    <SettingsGroup title="AI connection" label={`${expert.name} AI connection`}>
      <SettingsRow
        title="Connection"
        description={`Where ${expert.name}'s new threads, routines and follow-ups run. Existing threads keep theirs.`}
        control={
          <Select
            id={selectId}
            label="AI connection"
            hideLabel
            aria-describedby={noteId}
            size="small"
            value={value}
            options={options}
            disabled={isLoading || isSaving}
            onValueChange={selectRoute}
            wrapperClassName="!mb-0 w-full sm:w-64"
          />
        }
      >
        <Text
          id={noteId}
          variant="small"
          tone={note.warning ? undefined : "muted"}
          className={cn("mt-1", note.warning && "text-amber-700")}
        >
          {note.text}
        </Text>
      </SettingsRow>
      <SettingsRow
        title="Manage connections"
        description="Link or unlink ChatGPT and Microsoft 365 Copilot for your account."
        control={
          <Link href="/settings/integrations" className="whitespace-nowrap">
            Open Settings
          </Link>
        }
      />
    </SettingsGroup>
  );
}
