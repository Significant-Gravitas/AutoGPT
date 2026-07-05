"use client";

import { useGetV2GetSandboxChanges } from "@/app/api/__generated__/endpoints/chat/chat";

interface Props {
  sessionId: string;
}

export function ChangesCountBadge({ sessionId }: Props) {
  const { data: count } = useGetV2GetSandboxChanges(sessionId, {
    query: { select: (res) => res.data.files.length },
  });

  if (!count) return null;

  return (
    <span className="ml-1.5 inline-flex min-w-[1.25rem] items-center justify-center rounded-full bg-zinc-200 px-1.5 text-[0.6875rem] font-medium text-zinc-700">
      {count}
    </span>
  );
}
