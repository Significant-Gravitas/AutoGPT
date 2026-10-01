"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Link } from "@/components/atoms/Link/Link";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useExpertLlmRouteSection } from "./useExpertLlmRouteSection";

interface Props {
  expert: Expert;
}

export function ExpertLlmRouteSection({ expert }: Props) {
  const { value, options, note, isLoading, isSaving, selectRoute } =
    useExpertLlmRouteSection({ expert });

  return (
    <section
      aria-label={`${expert.name} AI connection`}
      className="flex w-full shrink-0 flex-col gap-1.5 md:w-64"
    >
      <div className="flex items-center justify-between gap-2">
        <Text variant="body-medium" tone="secondary">
          AI connection
        </Text>
        <Link href="/settings/integrations" variant="secondary">
          <Text variant="small" tone="secondary" as="span">
            Manage connections
          </Text>
        </Link>
      </div>
      <Select
        id={`expert-${expert.id}-llm-route`}
        label="AI connection"
        hideLabel
        size="small"
        value={value}
        options={options}
        disabled={isLoading || isSaving}
        onValueChange={selectRoute}
        wrapperClassName="!mb-0"
      />
      <Text
        variant="small"
        tone={note.warning ? undefined : "muted"}
        className={cn(note.warning && "text-amber-700")}
      >
        {note.text}
      </Text>
    </section>
  );
}
