"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Select } from "@/components/atoms/Select/Select";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { TeamBadge } from "@/components/contextual/TeamBadge/TeamBadge";
import { Dialog } from "@/components/molecules/Dialog/Dialog";

import {
  capabilityLabel,
  capabilityOptions,
  CredentialMode,
  credentialModeOptions,
} from "./helpers";
import { useShareAgentDialog } from "./useShareAgentDialog";

interface Props {
  agent: LibraryAgent;
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

export function ShareAgentDialog({ agent, isOpen, setIsOpen }: Props) {
  const {
    teams,
    teamId,
    setTeamId,
    capability,
    setCapability,
    followLatest,
    setFollowLatest,
    credentialMode,
    setCredentialMode,
    isOwner,
    grants,
    isLoadingGrants,
    isGrantsError,
    isSharing,
    canShare,
    handleShare,
    handleRevoke,
  } = useShareAgentDialog(agent, isOpen);

  return (
    <Dialog
      controlled={{ isOpen, set: setIsOpen }}
      styling={{ maxWidth: "34rem" }}
      title="Share agent"
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4" data-testid="share-agent-dialog">
          <Text variant="body" className="text-zinc-600">
            Share “{agent.name}” with a team in your organization.
          </Text>

          <Select
            id="share-team"
            label="Team"
            value={teamId ?? ""}
            onValueChange={(value) => setTeamId(value || null)}
            placeholder="Select a team"
            options={teams.map((team) => ({
              value: team.id,
              label: team.name,
            }))}
            size="small"
          />

          <Select
            id="share-capability"
            label="Access"
            value={capability}
            onValueChange={setCapability}
            options={capabilityOptions}
            size="small"
          />

          <div className="flex items-start justify-between gap-3">
            <div className="flex flex-col">
              <Text variant="body-medium">Always share latest version</Text>
              <Text variant="small" className="text-zinc-500">
                {followLatest
                  ? "The team always runs your newest version."
                  : `Pinned to the current version (v${agent.graph_version}).`}
              </Text>
            </div>
            <Switch
              checked={followLatest}
              onCheckedChange={setFollowLatest}
              aria-label="Always share latest version"
            />
          </div>

          {isOwner ? (
            <div className="flex flex-col gap-1">
              <Select
                id="share-credential-mode"
                label="Credentials"
                value={credentialMode}
                onValueChange={setCredentialMode}
                options={credentialModeOptions}
                size="small"
              />
              {credentialMode === CredentialMode.Owner ? (
                <Text variant="small" className="text-amber-600">
                  Runs use your connected accounts.
                </Text>
              ) : null}
            </div>
          ) : null}

          <div className="flex justify-end">
            <Button
              onClick={handleShare}
              loading={isSharing}
              disabled={!canShare}
            >
              Share
            </Button>
          </div>

          <div className="flex flex-col gap-2 border-t border-zinc-100 pt-4">
            <Text variant="body-medium">Shared with</Text>
            {isLoadingGrants ? (
              <Skeleton className="h-8 w-full" />
            ) : isGrantsError ? (
              <Text variant="small" className="text-red-600">
                Couldn’t load existing shares. Please try again.
              </Text>
            ) : grants.length === 0 ? (
              <Text variant="small" className="text-zinc-500">
                Not shared with any teams yet.
              </Text>
            ) : (
              <ul className="flex flex-col divide-y divide-zinc-100">
                {grants.map((grant) => (
                  <li
                    key={grant.id}
                    className="flex items-center justify-between gap-2 py-2"
                    data-testid="share-grant-row"
                  >
                    <div className="flex min-w-0 items-center gap-2">
                      <TeamBadge teamId={grant.principal_id} />
                      <Text variant="small" className="truncate text-zinc-500">
                        {capabilityLabel(grant.capability)} ·{" "}
                        {grant.follow_latest
                          ? "latest"
                          : `v${grant.agent_graph_version}`}
                      </Text>
                    </div>
                    <Button
                      variant="ghost"
                      size="small"
                      onClick={() => handleRevoke(grant.id)}
                    >
                      Revoke
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
