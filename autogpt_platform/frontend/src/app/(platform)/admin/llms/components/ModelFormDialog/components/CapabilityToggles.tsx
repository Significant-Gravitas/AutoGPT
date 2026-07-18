"use client";

import { Switch } from "@/components/atoms/Switch/Switch";
import type { ModelFormValues } from "../useModelFormDialog";

interface Props {
  values: ModelFormValues;
  setField: <K extends keyof ModelFormValues>(
    key: K,
    value: ModelFormValues[K],
  ) => void;
}

const CAPABILITIES: { key: keyof ModelFormValues; label: string }[] = [
  { key: "supports_tools", label: "Tools" },
  { key: "supports_json_output", label: "JSON output" },
  { key: "supports_reasoning", label: "Reasoning" },
  { key: "supports_parallel_tool_calls", label: "Parallel tool calls" },
];

const FLAGS: { key: keyof ModelFormValues; label: string }[] = [
  { key: "is_enabled", label: "Enabled" },
  { key: "is_recommended", label: "Recommended" },
];

export function CapabilityToggles({ values, setField }: Props) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm font-medium">Capabilities</p>
      <div className="grid grid-cols-2 gap-2">
        {CAPABILITIES.map(({ key, label }) => (
          <label key={key} className="flex items-center gap-2 text-sm">
            <Switch
              checked={Boolean(values[key])}
              onCheckedChange={(checked) => setField(key, checked)}
              aria-label={label}
            />
            {label}
          </label>
        ))}
      </div>
      <p className="text-sm font-medium">Flags</p>
      <div className="grid grid-cols-2 gap-2">
        {FLAGS.map(({ key, label }) => (
          <label key={key} className="flex items-center gap-2 text-sm">
            <Switch
              checked={Boolean(values[key])}
              onCheckedChange={(checked) => setField(key, checked)}
              aria-label={label}
            />
            {label}
          </label>
        ))}
      </div>
    </div>
  );
}
