"use client";

import Link from "next/link";
import {
  ArrowRight01Icon,
  Delete02Icon,
  Loading03Icon,
  PlugSocketIcon,
  PlusSignIcon,
  Share01Icon,
} from "@hugeicons/core-free-icons";
import type { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { ExpertConnectServiceDialog } from "@/app/(platform)/team/[expertId]/components/ExpertIntegrationsSection/ExpertConnectServiceDialog";
import {
  groupExpertIntegrations,
  TYPE_LABELS,
} from "@/app/(platform)/team/[expertId]/components/ExpertIntegrationsSection/ExpertIntegrationGroups";
import { UseExistingCredentialsDialog } from "@/app/(platform)/team/[expertId]/components/ExpertIntegrationsSection/UseExistingCredentialsDialog";
import { useExpertIntegrationsSection } from "@/app/(platform)/team/[expertId]/components/ExpertIntegrationsSection/useExpertIntegrationsSection";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { formatCredentialName } from "@/components/contextual/IntegrationsPanel/helpers";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { cn } from "@/lib/utils";
import { useCopilotUIStore, type ContextPanelExpert } from "../../../../store";
import { StackSection } from "../../../WorkspaceFileCards/components/StackSection";
import { PanelEmptyState } from "../PanelEmptyState/PanelEmptyState";

/** The chat's expert's integrations, laid out like the files tab: a titled
 *  card of rows, or an empty state that offers the first connection. Same
 *  add / use existing / remove actions as the expert's page. */
export function IntegrationsTab() {
  const expert = useCopilotUIStore((s) => s.contextPanelExpert);
  if (!expert) return null;

  return <IntegrationsTabContent expert={expert} />;
}

function IntegrationsTabContent({ expert }: { expert: ContextPanelExpert }) {
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
    isGrantableError,
    refetch,
  } = useExpertIntegrationsSection(expert.id);
  const groups = groupExpertIntegrations(granted);
  const canUseExisting = grantable.length > 0 && !isError && !isGrantableError;

  return (
    <div className="flex flex-col gap-6">
      {isGrantableError && !isError ? (
        <div className="flex items-center justify-between gap-3 rounded-xl bg-warning-surface px-3 py-2 ring-1 ring-inset ring-warning-border">
          <Text
            // TODO(tokens): no exact role for !text-amber-800
            variant="small"
            className="!text-amber-800"
          >
            Couldn&apos;t load your connected services, so &ldquo;Use
            existing&rdquo; is unavailable.
          </Text>
          <Button
            variant="secondary"
            size="small"
            onClick={() => void refetch()}
          >
            Retry
          </Button>
        </div>
      ) : null}
      {isLoading ? (
        <Skeleton className="h-24 w-full rounded-lg" />
      ) : isError ? (
        <ErrorCard
          context="this expert's integrations"
          hint="We could not load what this expert can reach."
          onRetry={refetch}
        />
      ) : groups.length === 0 ? (
        <PanelEmptyState
          tiles={[
            <IntegrationLogo key="slack" provider="slack" size={24} />,
            <IntegrationLogo key="notion" provider="notion" size={24} />,
            <IntegrationLogo key="google" provider="google" size={24} />,
          ]}
          title="No integrations yet"
          description={`Connect a tool and ${expert.name} can use it on your behalf.`}
          footer={<ExpertPageLink expert={expert} className="justify-center" />}
        >
          <Button
            variant="primary"
            size="small"
            leadingIcon={PlusSignIcon}
            onClick={openConnect}
          >
            Add integration
          </Button>
          {canUseExisting && (
            <Button
              variant="secondary"
              size="small"
              leadingIcon={Share01Icon}
              onClick={openUseExisting}
            >
              Use existing
            </Button>
          )}
        </PanelEmptyState>
      ) : (
        <StackSection
          title={`${expert.name}'s integrations`}
          icon={PlugSocketIcon}
          count={granted.length}
          action={
            <>
              {canUseExisting && (
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={openUseExisting}
                  aria-label="Use existing"
                  className="size-6 rounded-lg !p-0 text-foreground-subtle"
                >
                  <Icon icon={Share01Icon} size={14} />
                </Button>
              )}
              <Button
                variant="ghost"
                size="icon"
                onClick={openConnect}
                aria-label="Add integration"
                className="size-6 rounded-lg !p-0 text-foreground-subtle"
              >
                <Icon icon={PlusSignIcon} size={14} />
              </Button>
            </>
          }
        >
          <div className="-mx-3 grid divide-y divide-border-base/70">
            {groups.flatMap((group) =>
              group.integrations.map((integration) => (
                <IntegrationRow
                  key={integration.credential_id}
                  serviceName={group.name}
                  icon={group.icon}
                  integration={integration}
                  isRemoving={isRevoking}
                  onRemove={() => removeIntegration(integration.credential_id)}
                />
              )),
            )}
          </div>
          <div className="-mx-3 border-t border-border-base/70 px-3">
            <ExpertPageLink expert={expert} className="justify-between py-2" />
          </div>
        </StackSection>
      )}

      <UseExistingCredentialsDialog
        open={isUsingExisting}
        expertName={expert.name}
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
        expertName={expert.name}
        onClose={closeConnect}
        onConnected={(credential) => {
          connectCredential(credential);
          closeConnect();
        }}
      />
    </div>
  );
}

interface ExpertPageLinkProps {
  expert: ContextPanelExpert;
  className?: string;
}

function ExpertPageLink({ expert, className }: ExpertPageLinkProps) {
  return (
    <Link
      href={`/team/${expert.id}`}
      className={cn(
        "flex items-center gap-1 text-body-sm font-medium leading-inherit text-foreground-body transition-colors hover:text-foreground",
        className,
      )}
    >
      Open {expert.name}&apos;s page
      <Icon icon={ArrowRight01Icon} size={14} aria-hidden />
    </Link>
  );
}

interface RowProps {
  serviceName: string;
  icon: string;
  integration: ExpertCredentialRef;
  isRemoving: boolean;
  onRemove: () => void;
}

/** A files-tab style row: logo tile, service and account, and a remove action
 *  that fades in on hover (always shown where there is no hover). */
function IntegrationRow({
  serviceName,
  icon,
  integration,
  isRemoving,
  onRemove,
}: RowProps) {
  const name = formatCredentialName(integration.title, integration.provider);

  return (
    <div
      className="group flex items-center gap-3 px-3 py-2 transition-colors hover:bg-surface-hover/70"
      data-testid="expert-integration-row"
    >
      <span className="flex size-10 shrink-0 items-center justify-center rounded-xl border border-border-base bg-surface">
        <IntegrationLogo
          provider={icon}
          alt={`${serviceName} logo`}
          size={20}
        />
      </span>
      <span className="flex min-w-0 flex-1 flex-col">
        <Text
          variant="body-medium"
          as="span"
          className="truncate text-foreground-strong"
        >
          {serviceName}
        </Text>
        <Text
          variant="small"
          as="span"
          className="truncate text-foreground-tertiary"
        >
          {name} · {TYPE_LABELS[integration.type] ?? integration.type}
        </Text>
      </span>
      <span
        aria-hidden
        title="Ready"
        // TODO(tokens): no exact role for bg-emerald-500
        className="size-1.5 shrink-0 rounded-full bg-emerald-500"
      />
      <Button
        variant="ghost"
        size="icon"
        onClick={onRemove}
        disabled={isRemoving}
        aria-busy={isRemoving}
        aria-label={`Remove ${name}`}
        className="size-7 shrink-0 rounded-lg !p-0 text-foreground-tertiary opacity-0 transition-opacity hover:text-error focus-visible:opacity-100 group-hover:opacity-100 [@media(hover:none)]:opacity-100"
      >
        <Icon
          icon={isRemoving ? Loading03Icon : Delete02Icon}
          size={14}
          className={isRemoving ? "animate-spin" : undefined}
        />
      </Button>
    </div>
  );
}
