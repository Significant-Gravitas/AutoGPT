import { Text } from "@/components/atoms/Text/Text";
import { ArrowRightIcon, EnvelopeSimpleIcon } from "@phosphor-icons/react";
import type { TourArtifact } from "../../script/types";

interface Props {
  artifact: TourArtifact;
}

export function TourArtifactCard({ artifact }: Props) {
  return (
    <div className="flex flex-col gap-2.5 rounded-xl border border-zinc-200/70 bg-white p-4 shadow-sm">
      <div className="flex items-center gap-1.5 text-sm text-zinc-500">
        <EnvelopeSimpleIcon className="size-4 shrink-0" />
        <span>Artifact · {artifact.caption}</span>
      </div>

      <Text variant="large-medium" className="text-zinc-900">
        {artifact.title}
      </Text>

      {artifact.subtitle && (
        <Text variant="body" className="text-zinc-600">
          {artifact.subtitle}
        </Text>
      )}

      {artifact.bullets && artifact.bullets.length > 0 && (
        <ul className="flex list-disc flex-col gap-1.5 pl-5 text-sm text-zinc-700">
          {artifact.bullets.map((bullet) => (
            <li key={bullet}>{bullet}</li>
          ))}
        </ul>
      )}

      {artifact.diff && (
        <div className="flex items-center justify-between gap-3 rounded-lg bg-zinc-50 px-4 py-3">
          <div className="flex items-center gap-3 text-base font-medium">
            <span className="text-red-700 line-through decoration-red-700/70">
              {artifact.diff.from}
            </span>
            <ArrowRightIcon className="size-4 shrink-0 text-zinc-500" />
            <span className="text-emerald-700">{artifact.diff.to}</span>
          </div>
          <span className="text-sm text-zinc-500">{artifact.diff.delta}</span>
        </div>
      )}
    </div>
  );
}
