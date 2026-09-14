"use client";

import { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { lineDiff, stepChanges } from "@/services/skill-learning/helpers";
import { useState } from "react";

interface Props {
  version: SkillVersionSummary | null;
  versions: SkillVersionSummary[];
  onSelect: (versionId: string) => void;
}

export function ChangesView({ version, versions, onSelect }: Props) {
  const [showRaw, setShowRaw] = useState(false);
  if (!version) {
    return (
      <Text variant="small" tone="muted">
        Nothing to compare yet.
      </Text>
    );
  }
  const base = versions.find((item) => item.id === version.base_version_id);
  const before = base?.body ?? "";
  const after = version.body ?? "";
  const changes = stepChanges(before, after);
  return (
    <div className="flex flex-col gap-4">
      <label className="flex flex-col gap-1">
        <Text variant="small-medium" tone="primary">
          Version
        </Text>
        <select
          className="rounded-md border border-zinc-200 px-2 py-1 text-sm"
          value={version.id}
          onChange={(event) => onSelect(event.target.value)}
          aria-label="Choose a version"
        >
          {versions.map((item) => (
            <option key={item.id} value={item.id}>
              v{item.version} · {item.origin_label} · {item.state_label}
            </option>
          ))}
        </select>
      </label>
      <Text variant="small" tone="secondary">
        {base
          ? `Compared with v${base.version}`
          : "First version — nothing to compare with"}
      </Text>
      {changes.length === 0 ? (
        <Text variant="small" tone="muted">
          No step changed in this version.
        </Text>
      ) : (
        <ul className="flex flex-col gap-3" aria-label="Changes by step">
          {changes.map((change) => (
            <li key={change.step} className="rounded-lg bg-zinc-50 p-3">
              <Text variant="small-medium" tone="primary">
                {change.step}
              </Text>
              {change.before ? (
                <Text variant="small" tone="muted" className="line-through">
                  {change.before}
                </Text>
              ) : null}
              {change.after ? (
                <Text variant="small" tone="primary">
                  {change.after}
                </Text>
              ) : (
                <Text variant="small" tone="danger">
                  Removed
                </Text>
              )}
            </li>
          ))}
        </ul>
      )}
      <Button variant="ghost" size="small" onClick={() => setShowRaw(!showRaw)}>
        {showRaw ? "Hide raw diff" : "Show raw Markdown diff"}
      </Button>
      {showRaw ? (
        <pre className="overflow-x-auto rounded-lg bg-zinc-900 p-3 text-xs text-zinc-100">
          {lineDiff(before, after) || "(no line-level differences)"}
        </pre>
      ) : null}
    </div>
  );
}
