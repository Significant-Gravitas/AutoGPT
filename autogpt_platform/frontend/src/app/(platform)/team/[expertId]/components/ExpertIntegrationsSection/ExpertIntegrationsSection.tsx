"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import { useState } from "react";
import { ExpertConnectServiceDialog } from "./ExpertConnectServiceDialog";
import { ExpertIntegrationGroups } from "./ExpertIntegrationGroups";
import { UseExistingCredentialsDialog } from "./UseExistingCredentialsDialog";
import { PlusSignIcon, Share01Icon } from "@hugeicons/core-free-icons";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { useExpertIntegrationsSection } from "./useExpertIntegrationsSection";
import { ACTION_BUTTON_CLASS } from "@/app/(platform)/team/helpers";

interface Props {
  expertId: string;
  expertName: string;
}

export function ExpertIntegrationsSection({ expertId, expertName }: Props) {
  const {
    granted,
    grantable,
    isConnecting,
    isUsingExisting,
    openUseExisting,
    closeUseExisting,
    openConnect,
    closeConnect,
    connectCredential,
    grantExistingCredential,
    removeIntegration,
    isGranting,
    isRevoking,
    isLoading,
    isError,
    isGrantableLoading,
    isGrantableError,
    refetch,
  } = useExpertIntegrationsSection(expertId);
  const [query, setQuery] = useState("");
  const needle = query.trim().toLowerCase();
  const visible = needle
    ? granted.filter((integration) =>
        `${integration.title} ${integration.provider}`
          .toLowerCase()
          .includes(needle),
      )
    : granted;

  return (
    <section data-testid="expert-integrations-section">
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-3">
        <Text variant="large-medium">{expertName}&apos;s Integrations</Text>
        <div className="flex items-center gap-2">
          <Tooltip>
            <TooltipTrigger asChild>
              <span>
                <Button
                  variant="secondary"
                  size="small"
                  className={ACTION_BUTTON_CLASS}
                  leftIcon={<Icon icon={Share01Icon} size={14} />}
                  disabled={
                    grantable.length === 0 || isError || isGrantableError
                  }
                  onClick={openUseExisting}
                >
                  Use existing
                </Button>
              </span>
            </TooltipTrigger>
            {isError ? (
              <TooltipContent side="bottom">
                Couldn&apos;t load what {expertName} already has.
              </TooltipContent>
            ) : isGrantableError ? (
              <TooltipContent side="bottom">
                Couldn&apos;t load your services.
              </TooltipContent>
            ) : grantable.length === 0 && !isGrantableLoading ? (
              <TooltipContent side="bottom">
                {expertName} can already use everything in your account.
              </TooltipContent>
            ) : null}
          </Tooltip>
          <Button
            variant="secondary"
            size="small"
            className={ACTION_BUTTON_CLASS}
            leftIcon={<Icon icon={PlusSignIcon} size={14} />}
            onClick={openConnect}
          >
            Add integration
          </Button>
          <SearchInput
            size="xsmall"
            value={query}
            onChange={setQuery}
            placeholder="Search integrations"
            className="w-48"
          />
        </div>
      </div>

      {isLoading ? (
        <Skeleton className="h-24 w-full rounded-lg" />
      ) : isError ? (
        <ErrorCard
          context="this expert's integrations"
          hint="We could not load what this expert can reach."
          onRetry={refetch}
        />
      ) : granted.length === 0 ? (
        <p className="text-sm text-zinc-500">
          Nothing connected yet. Add a tool and {expertName} can use it on your
          behalf.
        </p>
      ) : visible.length === 0 ? (
        <p className="pt-4 text-sm text-zinc-500">No integrations match.</p>
      ) : (
        <ExpertIntegrationGroups
          integrations={visible}
          isRemoving={isRevoking}
          onRemove={removeIntegration}
        />
      )}

      <UseExistingCredentialsDialog
        open={isUsingExisting}
        expertName={expertName}
        credentials={grantable}
        isGranting={isGranting}
        onUse={(credentialId) => {
          grantExistingCredential(credentialId);
          closeUseExisting();
        }}
        onClose={closeUseExisting}
      />

      <ExpertConnectServiceDialog
        open={isConnecting}
        expertName={expertName}
        onClose={closeConnect}
        onConnected={(credential) => {
          connectCredential(credential);
          closeConnect();
        }}
      />
    </section>
  );
}
