"use client";

import { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { lineDiff, stepChanges } from "@/services/skill-learning/helpers";
import { useState } from "react";
import { PackageFiles } from "./PackageFiles/PackageFiles";

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
    <div
      data-slot="skill-learning-sheet-changes"
      className="flex flex-col gap-4"
    >
      <label
        data-slot="skill-learning-sheet-changes-version-field"
        className="flex flex-col gap-1"
      >
        <Text variant="small-medium" tone="primary">
          Version
        </Text>
        <select
          data-slot="skill-learning-sheet-changes-version-picker"
          className="rounded-md border border-border-base px-2 py-1 text-sm"
          value={version.id}
          onChange={(event) => onSelect(event.target.value)}
          aria-label="Choose a version"
        >
          {versions.map((item) => (
            <option
              key={item.id}
              data-slot="skill-learning-sheet-changes-version-option"
              value={item.id}
            >
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
        <ul
          data-slot="skill-learning-sheet-changes-list"
          className="flex flex-col gap-3"
          aria-label="Changes by step"
        >
          {changes.map((change) => (
            <li
              key={change.step}
              data-slot="skill-learning-sheet-changes-item"
              className="rounded-lg bg-surface-muted p-3"
            >
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
      <PackageFiles before={base?.files} after={version.files} />
      {showRaw ? (
        <pre
          data-slot="skill-learning-sheet-changes-raw-diff"
          // TODO(tokens): no foreground role for zinc-100 (#EFEFF0) on the dark
          // code surface.
          className="overflow-x-auto rounded-lg bg-action-hover p-3 text-xs text-zinc-100"
        >
          {lineDiff(before, after) || "(no line-level differences)"}
        </pre>
      ) : null}
    </div>
  );
}
