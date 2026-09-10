"use client";

import { formatWhen } from "@/app/(platform)/settings/memory/helpers";
import type { MemoryFact } from "@/app/api/__generated__/models/memoryFact";
import { Button } from "@/components/atoms/Button/Button";
import { Link } from "@/components/atoms/Link/Link";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { SoulSectionTitle } from "../../SoulSectionTitle";
import { useLearnedNotes } from "./useLearnedNotes";

interface Props {
  expertId: string;
}

export function LearnedNotes({ expertId }: Props) {
  const { facts, isLoading, isError, forget, forgettingUuid } =
    useLearnedNotes(expertId);

  return (
    <section className="mb-8">
      <SoulSectionTitle>What I&apos;ve learned</SoulSectionTitle>
      {isLoading ? (
        <NotesSkeleton />
      ) : isError ? (
        <Text variant="small" tone="muted">
          Couldn&apos;t load what this expert has learned.
        </Text>
      ) : facts.length === 0 ? (
        <Text variant="small" tone="muted">
          Nothing recorded yet. What this expert learns will appear here.
        </Text>
      ) : (
        <NotesList
          expertId={expertId}
          facts={facts}
          forgettingUuid={forgettingUuid}
          onForget={forget}
        />
      )}
    </section>
  );
}

interface NotesListProps {
  expertId: string;
  facts: MemoryFact[];
  forgettingUuid: string | null;
  onForget: (uuid: string) => void;
}

function NotesList({
  expertId,
  facts,
  forgettingUuid,
  onForget,
}: NotesListProps) {
  return (
    <>
      <Text variant="small" tone="muted">
        Picked up from your conversations. Forget anything that shouldn&apos;t
        stick.
      </Text>
      <div className="mt-2 flex flex-col divide-y divide-zinc-100">
        {facts.map((fact) => (
          <div
            key={fact.uuid}
            className="flex items-start justify-between gap-2 py-2.5"
          >
            <div className="min-w-0 flex-1">
              <Text
                variant="small"
                as="p"
                unmask={false}
                className="text-textBlack"
              >
                {fact.fact || `${fact.source} → ${fact.target}`}
              </Text>
              <Text
                variant="small"
                as="span"
                unmask={false}
                className="text-zinc-400"
              >
                {formatWhen(fact.created_at)}
              </Text>
            </div>
            <Button
              variant="ghost"
              size="small"
              className="h-7 !min-w-0 shrink-0 px-2 text-zinc-600"
              loading={forgettingUuid === fact.uuid}
              onClick={() => onForget(fact.uuid)}
            >
              Forget
            </Button>
          </div>
        ))}
      </div>
      <Link
        href={`/settings/memory?expert=${encodeURIComponent(expertId)}`}
        variant="secondary"
        className="mt-3 inline-block text-zinc-600"
      >
        See all in memory settings
      </Link>
    </>
  );
}

function NotesSkeleton() {
  return (
    <div className="mt-2 flex flex-col gap-2 py-1">
      <Skeleton className="h-5 w-3/4" />
      <Skeleton className="h-5 w-2/3" />
    </div>
  );
}
