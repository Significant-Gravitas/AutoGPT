"use client";

import { ExpertLearnedNote } from "@/app/api/__generated__/models/expertLearnedNote";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { ReactNode } from "react";
import { formatLearnedAt } from "./helpers";
import { useLearnedNotes } from "./useLearnedNotes";

interface Props {
  expertId: string | undefined;
  title: ReactNode;
}

export function LearnedNotes({ expertId, title }: Props) {
  const {
    isFeatureEnabled,
    notes,
    isLoading,
    isError,
    deletingNoteId,
    forgetNote,
  } = useLearnedNotes({ expertId });

  if (!isFeatureEnabled) return null;

  return (
    <section className="mb-8">
      {title}
      {isLoading ? (
        <div className="mt-3 space-y-2">
          <Skeleton className="h-10 w-full rounded-xl" />
          <Skeleton className="h-10 w-3/4 rounded-xl" />
        </div>
      ) : null}
      {!isLoading && isError ? (
        <Text variant="small" className="text-zinc-500">
          We couldn&apos;t load what this expert has learned.
        </Text>
      ) : null}
      {!isLoading && !isError && notes.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          Nothing recorded yet. What this expert learns will appear here.
        </Text>
      ) : null}
      {notes.length > 0 ? (
        <ul className="mt-3 space-y-2">
          {notes.map((note) => (
            <LearnedNoteRow
              key={note.id}
              note={note}
              isDeleting={deletingNoteId === note.id}
              onForget={() => forgetNote(note.id)}
            />
          ))}
        </ul>
      ) : null}
    </section>
  );
}

interface RowProps {
  note: ExpertLearnedNote;
  isDeleting: boolean;
  onForget: () => void;
}

function LearnedNoteRow({ note, isDeleting, onForget }: RowProps) {
  return (
    <li className="flex items-start gap-3 rounded-xl bg-zinc-50 p-3">
      <div className="min-w-0 flex-1">
        <Text variant="small" className="text-zinc-700">
          {note.text}
        </Text>
        <Text variant="small" className="mt-1 text-zinc-400">
          {`learned ${formatLearnedAt(note.learned_at)}`}
        </Text>
      </div>
      <Button
        type="button"
        variant="icon"
        size="icon"
        loading={isDeleting}
        aria-label={`Forget: ${note.text}`}
        onClick={onForget}
      >
        <Icon icon={Delete02Icon} size={16} />
      </Button>
    </li>
  );
}
