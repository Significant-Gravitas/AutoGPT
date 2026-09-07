"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ConnectServiceDialog } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/ConnectServiceDialog";
import {
  CheckmarkCircle02Icon,
  PlugSocketIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import { useInstallSkillPanel } from "./useInstallSkillPanel";

interface Props {
  slug: string;
  requiredProviders: string[];
}

export function InstallSkillPanel({ slug, requiredProviders }: Props) {
  const {
    isLoggedIn,
    providerNames,
    installedName,
    isInstalling,
    addToAutoPilot,
    pendingConnections,
    isConnectOpen,
    openConnect,
    setIsConnectOpen,
    handleConnected,
  } = useInstallSkillPanel({ slug, requiredProviders });

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-zinc-200/80 bg-zinc-50/60 p-5">
      {installedName ? (
        <div
          className="flex flex-wrap items-center gap-2"
          data-testid="skill-installed"
        >
          <Icon
            icon={CheckmarkCircle02Icon}
            size={18}
            className="text-emerald-600"
          />
          <Text variant="body-medium">Added to your AutoPilot</Text>
          <Link
            href="/library/skills"
            className="text-sm font-medium text-accent hover:underline"
          >
            View in your skills
          </Link>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-3">
          {isLoggedIn ? (
            <Button
              variant="primary"
              onClick={addToAutoPilot}
              loading={isInstalling}
              data-testid="skill-install-button"
            >
              Add to AutoPilot
            </Button>
          ) : (
            <Button variant="primary" as="NextLink" href="/login">
              Add to AutoPilot
            </Button>
          )}
          {requiredProviders.length > 0 ? (
            <Text variant="small" className="!text-zinc-500">
              Works with {providerNames.join(", ")}
            </Text>
          ) : null}
        </div>
      )}

      {installedName && pendingConnections.length > 0 ? (
        <ConnectStep
          names={pendingConnections.map((provider) => provider.name)}
          onConnect={openConnect}
        />
      ) : null}

      <ConnectServiceDialog
        open={isConnectOpen}
        onOpenChange={setIsConnectOpen}
        onConnected={handleConnected}
      />
    </div>
  );
}

interface ConnectStepProps {
  names: string[];
  onConnect: () => void;
}

/** Offered after the install has already succeeded, so it reads as the next
 *  thing worth doing rather than something that went wrong. */
function ConnectStep({ names, onConnect }: ConnectStepProps) {
  return (
    <div
      className="flex flex-wrap items-center gap-3 rounded-xl bg-white p-4"
      data-testid="skill-connect-step"
    >
      <Icon icon={PlugSocketIcon} size={18} className="text-zinc-500" />
      <Text variant="small" className="!text-zinc-600">
        Its steps use {names.join(" and ")}. Connect when you first need it.
      </Text>
      <Button
        variant="secondary"
        size="small"
        onClick={onConnect}
        className="ml-auto"
      >
        Connect
      </Button>
    </div>
  );
}
