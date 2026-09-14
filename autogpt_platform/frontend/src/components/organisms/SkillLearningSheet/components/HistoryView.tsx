"use client";

import { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { canRestore } from "@/services/skill-learning/helpers";
import { formatDistanceToNow } from "date-fns";

interface Props {
  versions: SkillVersionSummary[];
  currentVersionId: string | null;
  isBusy: boolean;
  onRestore: (versionId: string) => void;
  onSelect: (versionId: string) => void;
}

export function HistoryView({
  versions,
  currentVersionId,
  isBusy,
  onRestore,
  onSelect,
}: Props) {
  return (
    <div className="flex flex-col gap-3">
      <Text variant="small" tone="muted">
        Restoring affects future use only and cannot undo actions already taken.
        The current version stays in history.
      </Text>
      <ul className="flex flex-col gap-2" aria-label="Version history">
        {versions.map((version) => {
          const restorable = canRestore(version, currentVersionId);
          return (
            <li
              key={version.id}
              className="flex flex-col gap-1 rounded-lg bg-zinc-50 p-3 sm:flex-row sm:items-center sm:justify-between"
            >
              <div className="min-w-0">
                <Text variant="small-medium" tone="primary">
                  v{version.version} · {version.origin_label}
                  {version.id === currentVersionId ? " · current" : ""}
                </Text>
                <Text variant="small" tone="muted">
                  {version.state_label} ·{" "}
                  {formatDistanceToNow(new Date(version.created_at), {
                    addSuffix: true,
                  })}
                  {version.summary ? ` · ${version.summary}` : ""}
                </Text>
              </div>
              <div className="flex shrink-0 gap-2">
                <Button
                  variant="ghost"
                  size="small"
                  onClick={() => onSelect(version.id)}
                >
                  View
                </Button>
                {restorable ? (
                  <Button
                    variant="secondary"
                    size="small"
                    disabled={isBusy}
                    onClick={() => onRestore(version.id)}
                    aria-label={`Restore v${version.version}`}
                  >
                    Restore
                  </Button>
                ) : null}
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
