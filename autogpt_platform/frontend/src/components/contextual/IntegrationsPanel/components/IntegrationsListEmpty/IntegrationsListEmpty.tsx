"use client";

import { Text } from "@/components/atoms/Text/Text";
import { IntegrationsMarquee } from "@/components/molecules/IntegrationsMarquee/IntegrationsMarquee";

interface Props {
  query: string;
}

export function IntegrationsListEmpty({ query }: Props) {
  const trimmed = query.trim();

  // A search usually lands in Available integrations below, so a miss here
  // is one quiet line rather than the full empty state.
  if (trimmed) {
    return (
      <Text variant="body" className="px-4 text-zinc-500">
        {`None of your connected integrations match "${trimmed}".`}
      </Text>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center gap-4 px-6 py-10 text-center">
      <IntegrationsMarquee />
      <div className="flex flex-col items-center gap-1">
        <Text variant="large-medium" as="span" className="text-textBlack">
          No integration connected
        </Text>
        <Text variant="body" className="max-w-[360px] text-zinc-500">
          Connect a service to let your agents use third-party tools like
          GitHub, Gmail, or Figma.
        </Text>
      </div>
    </div>
  );
}
