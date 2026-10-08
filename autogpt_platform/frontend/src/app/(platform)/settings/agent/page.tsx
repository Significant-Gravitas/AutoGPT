"use client";

import { RadioGroup } from "@/components/atoms/RadioGroup/RadioGroup";
import { Text } from "@/components/atoms/Text/Text";
import { useCommunicationMode } from "@/hooks/useCommunicationMode";
import { useEffect } from "react";

const MODE_OPTIONS = [
  {
    value: "compact",
    label: "Compact",
    description:
      "Friendly replies in plain prose. The tools and steps stay out of the way.",
  },
  {
    value: "technical",
    label: "Technical",
    description:
      "The detailed view: every tool call, step and result as it happens.",
  },
];

export default function SettingsAgentPage() {
  const { mode, setMode } = useCommunicationMode();

  useEffect(() => {
    document.title = "Agent – AutoGPT Platform";
  }, []);

  return (
    <div className="flex flex-col gap-4">
      <header className="flex min-w-0 flex-col pb-2 pl-4">
        <Text variant="h4" as="h1">
          Agent
        </Text>
        <Text variant="body" tone="secondary" className="mt-4 max-w-[600px]">
          How your agent talks to you.
        </Text>
      </header>
      <section className="rounded-2xl border border-border bg-card px-4 py-4 shadow-xs">
        <RadioGroup
          label="How do you want your agent to communicate?"
          options={MODE_OPTIONS}
          value={mode}
          onValueChange={(next) =>
            setMode(next === "compact" ? "compact" : "technical")
          }
        />
      </section>
    </div>
  );
}
