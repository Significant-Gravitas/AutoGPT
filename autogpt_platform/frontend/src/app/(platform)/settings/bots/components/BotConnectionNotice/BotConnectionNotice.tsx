"use client";

import Link from "next/link";

import { useGetV2ListChatTransports } from "@/app/api/__generated__/endpoints/chat/chat";
import { Text } from "@/components/atoms/Text/Text";

/**
 * A conversation that arrives over a bot link has no picker in front of it, so
 * the connection it starts on is whatever was chosen in Settings. That is the
 * least discoverable consequence of the setting — say it here, where someone
 * is looking at their bots.
 */
export function BotConnectionNotice() {
  const transportsQuery = useGetV2ListChatTransports();
  const transports =
    transportsQuery.data?.status === 200
      ? transportsQuery.data.data.transports
      : [];
  const current = transports.find((transport) => transport.default);

  if (!current || transports.filter((t) => t.available).length < 2) return null;

  return (
    <div className="mb-6 ml-4 rounded-2xl border border-[#DADADC] bg-white px-4 py-3">
      <Text variant="small" className="text-[#505057]">
        Conversations that come in over a bot start on{" "}
        <span className="font-medium text-black">{current.label}</span>, the
        connection you chose in{" "}
        <Link
          href="/settings/integrations"
          className="font-medium text-[#7444E5] underline underline-offset-2"
        >
          AI subscriptions
        </Link>
        .
      </Text>
    </div>
  );
}
