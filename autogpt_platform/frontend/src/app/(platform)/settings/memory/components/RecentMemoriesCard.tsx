"use client";

import type { MemoryFact } from "@/app/api/__generated__/models/memoryFact";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { formatWhen } from "../helpers";

interface Props {
  facts: MemoryFact[];
  isLoading: boolean;
  forgettingUuid: string | null;
  onForget: (uuid: string) => void;
  onForgetTopic: () => void;
}

export function RecentMemoriesCard({
  facts,
  isLoading,
  forgettingUuid,
  onForget,
  onForgetTopic,
}: Props) {
  return (
    <div className="flex flex-col rounded-[18px] border border-zinc-200 bg-white px-4 py-4 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
      <Text variant="body-medium" as="span" className="text-textBlack">
        Recent memories
      </Text>
      <Text variant="small" as="span" className="text-zinc-500">
        The latest things this memory has learned. Forget any of them.
      </Text>

      <div className="mt-2 flex flex-col divide-y divide-zinc-100">
        {isLoading ? (
          <div className="flex flex-col gap-2 py-3">
            <Skeleton className="h-5 w-3/4" />
            <Skeleton className="h-5 w-2/3" />
            <Skeleton className="h-5 w-4/5" />
          </div>
        ) : facts.length === 0 ? (
          <Text variant="small" as="p" className="py-3 text-zinc-500">
            Nothing remembered yet. Memories show up here as you chat.
          </Text>
        ) : (
          facts.map((fact) => (
            <div
              key={fact.uuid}
              className="flex items-center justify-between gap-3 py-2.5"
            >
              <Text
                variant="small"
                as="span"
                unmask={false}
                className="min-w-0 flex-1 text-textBlack"
              >
                {fact.fact || `${fact.source} → ${fact.target}`}
              </Text>
              <Text
                variant="small"
                as="span"
                unmask={false}
                className="hidden shrink-0 text-zinc-400 sm:inline"
              >
                {formatWhen(fact.created_at)}
              </Text>
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
          ))
        )}
      </div>

      <div className="mt-3">
        <Button variant="secondary" size="small" onClick={onForgetTopic}>
          Forget a topic…
        </Button>
      </div>
    </div>
  );
}
