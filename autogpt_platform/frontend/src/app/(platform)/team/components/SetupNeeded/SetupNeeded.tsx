"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Alert01Icon } from "@hugeicons/core-free-icons";
import { ExpertConnectServiceDialog } from "../../[expertId]/components/ExpertIntegrationsSection/ExpertConnectServiceDialog";
import { SetupNeededRow } from "./components/SetupNeededRow";
import { useSetupNeeded } from "./useSetupNeeded";

interface Props {
  enabled: boolean;
}

/** Everything still standing between the team's scheduled workflows and
 *  their schedules, one row per gap with the one action that closes it. */
export function SetupNeeded({ enabled }: Props) {
  const {
    items,
    isConnectable,
    connecting,
    connect,
    closeConnect,
    handleConnected,
    allow,
    isGranting,
  } = useSetupNeeded({ enabled });

  if (items.length === 0) return null;

  return (
    <section
      aria-label="Setup needed"
      className="flex min-w-0 flex-col"
      data-testid="setup-needed"
    >
      <div className="mb-1.5 flex items-center gap-1.5 px-3.5">
        <Icon
          icon={Alert01Icon}
          size={14}
          className="text-amber-600"
          aria-hidden
        />
        <Text variant="small-medium" as="h2" className="!text-zinc-700">
          Setup needed ({items.length})
        </Text>
      </div>
      <ul className="flex flex-col gap-2" aria-label="Setup items">
        {items.map((item) => (
          <li key={`${item.workflow_id}-${item.providers.join("+")}`}>
            <SetupNeededRow
              item={item}
              isConnectable={isConnectable(item)}
              isGranting={isGranting}
              onConnect={() => connect(item)}
              onAllow={() => allow(item)}
            />
          </li>
        ))}
      </ul>
      {connecting ? (
        <ExpertConnectServiceDialog
          open
          expertName={connecting.expert_name}
          initialProviderId={connecting.providers[0] ?? null}
          onClose={closeConnect}
          onConnected={handleConnected}
        />
      ) : null}
    </section>
  );
}
