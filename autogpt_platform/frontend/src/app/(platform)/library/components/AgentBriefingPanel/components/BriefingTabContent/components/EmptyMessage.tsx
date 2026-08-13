"use client";

import { Text } from "@/components/atoms/Text/Text";
import type { AgentStatusFilter } from "@/app/(platform)/library/types";
import { getEmptyMessage } from "../helpers";

interface Props {
  tab: AgentStatusFilter;
}

export function EmptyMessage({ tab }: Props) {
  return (
    <div className="flex items-center justify-center pt-4">
      <Text variant="body-medium" className="text-zinc-600">
        {getEmptyMessage(tab)}
      </Text>
    </div>
  );
}
