"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Text } from "@/components/atoms/Text/Text";
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
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { SectionEmptyState } from "../../../components/SectionEmptyState";
import { ToolbarSearch } from "../../../components/ToolbarSearch";

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
        <Text variant="large-medium" tone="primary">
          {expertName}&apos;s Integrations
        </Text>
        <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
          <Tooltip>
            <TooltipTrigger asChild>
              <span>
                <Button
                  variant="secondary"
                  size="small"
                  leadingIcon={Share01Icon}
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
            leadingIcon={PlusSignIcon}
            onClick={openConnect}
          >
            Add integration
          </Button>
          <ToolbarSearch
            id="expert-integrations-search"
            label="Search integrations"
            value={query}
            onChange={setQuery}
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
        <SectionEmptyState
          tiles={[
            <IntegrationLogo key="slack" provider="slack" size={24} />,
            <IntegrationLogo key="notion" provider="notion" size={24} />,
            <IntegrationLogo key="google" provider="google" size={24} />,
          ]}
          title="No integrations yet"
          description={`Connect a tool and ${expertName} can use it on your behalf.`}
        >
          <Button
            variant="primary"
            size="small"
            leadingIcon={PlusSignIcon}
            onClick={openConnect}
          >
            Add integration
          </Button>
          {grantable.length > 0 && !isGrantableError ? (
            <Button
              variant="secondary"
              size="small"
              leadingIcon={Share01Icon}
              onClick={openUseExisting}
            >
              Use existing
            </Button>
          ) : null}
        </SectionEmptyState>
      ) : visible.length === 0 ? (
        <Text variant="body" tone="muted" className="pt-4">
          No integrations match.
        </Text>
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
