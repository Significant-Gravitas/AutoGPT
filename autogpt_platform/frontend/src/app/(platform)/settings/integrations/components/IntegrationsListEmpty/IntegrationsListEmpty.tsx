"use client";

import { Text } from "@/components/atoms/Text/Text";

import { IntegrationsMarquee } from "./components/IntegrationsMarquee";

interface Props {
  query: string;
}

export function IntegrationsListEmpty({ query }: Props) {
  const hasQuery = query.trim().length > 0;

  return (
    <div className="flex flex-col items-center justify-center gap-4 px-6 py-10 text-center">
      <IntegrationsMarquee />
      <div className="flex flex-col items-center gap-1">
        <Text variant="large-medium" as="span" className="text-textBlack">
          {hasQuery ? "No integrations found" : "No integration connected"}
        </Text>
        <Text variant="body" className="max-w-[360px] text-zinc-500">
          {hasQuery
            ? `No integrations match "${query.trim()}". Try a different search.`
            : "Connect a service to let your agents use third-party tools like GitHub, Gmail, or Figma."}
        </Text>
      </div>
    </div>
  );
}
