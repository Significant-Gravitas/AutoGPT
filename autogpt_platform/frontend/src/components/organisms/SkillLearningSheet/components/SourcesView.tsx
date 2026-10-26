"use client";

import { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import Link from "next/link";

interface Props {
  version: SkillVersionSummary | null;
  isBusy: boolean;
  onExclude: (sourceKind: string, sourceRef: string) => void;
}

export function SourcesView({ version, isBusy, onExclude }: Props) {
  const sources = version?.sources ?? [];
  if (!version || sources.length === 0) {
    return (
      <Text variant="small" tone="muted">
        No sources are attached to this version.
      </Text>
    );
  }
  return (
    <ul
      data-slot="skill-learning-sheet-sources"
      className="flex flex-col gap-3"
      aria-label="Sources"
    >
      {sources.map((source) => (
        <li
          key={source.source_id}
          data-slot="skill-learning-sheet-sources-item"
          className="rounded-lg bg-surface-muted p-3"
        >
          {source.accessible ? (
            <>
              <Text variant="small-medium" tone="primary">
                {source.title || "Conversation"}
              </Text>
              <Text variant="small" tone="muted">
                Revision {source.revision.replace(/^0+/, "") || "0"}
                {source.excluded ? " · Excluded from learning" : ""}
              </Text>
              <div
                data-slot="skill-learning-sheet-sources-item-actions"
                className="mt-2 flex flex-wrap gap-2"
              >
                {source.url ? (
                  <Link
                    href={source.url}
                    target="_blank"
                    rel="noreferrer"
                    data-slot="skill-learning-sheet-sources-item-link"
                    className="text-sm text-accent-strong underline"
                  >
                    Open source
                  </Link>
                ) : null}
                {!source.excluded && source.source_ref ? (
                  <Button
                    variant="ghost"
                    size="small"
                    disabled={isBusy}
                    onClick={() =>
                      onExclude(source.source_kind, source.source_ref ?? "")
                    }
                  >
                    Exclude this source from learning
                  </Button>
                ) : null}
              </div>
            </>
          ) : (
            <Text variant="small" tone="muted">
              Evidence not visible to you
            </Text>
          )}
        </li>
      ))}
    </ul>
  );
}
