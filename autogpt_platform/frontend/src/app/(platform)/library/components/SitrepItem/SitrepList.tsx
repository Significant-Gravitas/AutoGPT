"use client";

import { Text } from "@/components/atoms/Text/Text";
import { useSitrepItems } from "./useSitrepItems";
import { SitrepItem } from "./SitrepItem";
import { useAutoPilotBridge } from "@/contexts/AutoPilotBridgeContext";

interface Props {
  agentIDs: string[];
  maxItems?: number;
}

export function SitrepList({ agentIDs, maxItems = 10 }: Props) {
  const items = useSitrepItems(agentIDs, maxItems);
  const { sendPrompt } = useAutoPilotBridge();

  if (items.length === 0) {
    return (
      <div className="py-4 text-center">
        <Text variant="small" className="text-zinc-400">
          All agents are healthy — nothing to report.
        </Text>
      </div>
    );
  }

  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <Text variant="small-medium" className="text-zinc-700">
          AI Summary
        </Text>
        <Text variant="xsmall" className="text-zinc-400">
          Updated just now
        </Text>
      </div>
      <div className="space-y-1">
        {items.map((item) => (
          <SitrepItem
            key={item.id}
            item={item}
            onAskAutoPilot={sendPrompt}
          />
        ))}
      </div>
    </div>
  );
}
