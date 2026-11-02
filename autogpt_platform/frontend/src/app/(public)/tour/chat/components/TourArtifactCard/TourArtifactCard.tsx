import { Text } from "@/components/atoms/Text/Text";
import type { TourArtifact } from "../../script/types";
import { ArrowRight02Icon, Mail02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  artifact: TourArtifact;
}

export function TourArtifactCard({ artifact }: Props) {
  return (
    <div className="flex flex-col gap-2.5 rounded-xl border border-border-base/70 bg-surface p-4 shadow-sm">
      <div className="flex items-center gap-1.5 text-sm text-foreground-tertiary">
        <Icon icon={Mail02Icon} className="size-4 shrink-0" />
        <span>Artifact · {artifact.caption}</span>
      </div>

      <Text variant="large-medium" tone="primary">
        {artifact.title}
      </Text>

      {artifact.subtitle && (
        <Text variant="body" tone="secondary">
          {artifact.subtitle}
        </Text>
      )}

      {artifact.bullets && artifact.bullets.length > 0 && (
        <ul className="flex list-disc flex-col gap-1.5 pl-5 text-sm text-foreground-body">
          {artifact.bullets.map((bullet) => (
            <li key={bullet}>{bullet}</li>
          ))}
        </ul>
      )}

      {artifact.diff && (
        <div className="flex items-center justify-between gap-3 rounded-lg bg-surface-muted px-4 py-3">
          <div className="flex items-center gap-3 text-base font-medium">
            <span className="text-error-foreground line-through decoration-error-foreground/70">
              {artifact.diff.from}
            </span>
            <span className="contents text-foreground-tertiary">
              <Icon icon={ArrowRight02Icon} className="size-4 shrink-0" />
            </span>
            <span className="text-success-foreground">{artifact.diff.to}</span>
          </div>
          <span className="text-sm text-foreground-tertiary">
            {artifact.diff.delta}
          </span>
        </div>
      )}
    </div>
  );
}
