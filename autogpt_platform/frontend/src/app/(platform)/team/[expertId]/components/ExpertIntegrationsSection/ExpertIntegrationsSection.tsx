"use client";

import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ConnectServiceDialog } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/ConnectServiceDialog";
import {
  formatCredentialName,
  formatCredentialSource,
  formatProviderName,
} from "@/components/contextual/IntegrationsPanel/helpers";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Delete02Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { useExpertIntegrationsSection } from "./useExpertIntegrationsSection";

interface Props {
  expertId: string;
  expertName: string;
  expertAvatarUrl: string | null;
}

export function ExpertIntegrationsSection({
  expertId,
  expertName,
  expertAvatarUrl,
}: Props) {
  const {
    granted,
    grantable,
    isAdding,
    openAdd,
    closeAdd,
    isConnecting,
    openConnect,
    closeConnect,
    connectCredential,
    addIntegration,
    removeIntegration,
    isGranting,
    isRevoking,
    isLoading,
    isError,
    isGrantableLoading,
    isGrantableError,
    refetch,
  } = useExpertIntegrationsSection(expertId);

  return (
    <section data-testid="expert-integrations-section">
      <div className="mb-2.5 flex items-center justify-between">
        <div className="text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
          Integrations
        </div>
        <Popover
          open={isAdding}
          onOpenChange={(open) => (open ? openAdd() : closeAdd())}
        >
          <PopoverTrigger asChild>
            <Button
              variant="ghost"
              size="small"
              leftIcon={<Icon icon={PlusSignIcon} size={16} />}
            >
              Add integration
            </Button>
          </PopoverTrigger>
          <PopoverContent align="end" className="w-72 p-2">
            {isGrantableLoading ? (
              <div className="flex flex-col gap-2 p-2">
                <Skeleton className="h-8 w-full" />
                <Skeleton className="h-8 w-full" />
              </div>
            ) : isGrantableError ? (
              <p className="px-2 py-3 text-sm text-zinc-500">
                Couldn&apos;t load your services.
              </p>
            ) : isError ? (
              <p className="px-2 py-3 text-sm text-zinc-500">
                Couldn&apos;t load what {expertName} already has.
              </p>
            ) : grantable.length === 0 ? (
              <p className="px-2 py-3 text-sm text-zinc-500">
                {granted.length === 0
                  ? "Nothing connected yet."
                  : `${expertName} can already use everything you've connected.`}
              </p>
            ) : (
              <ul className="max-h-72 overflow-y-auto">
                {grantable.map((credential) => (
                  <li key={credential.id}>
                    <button
                      type="button"
                      disabled={isGranting}
                      className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-left hover:bg-zinc-50 disabled:opacity-50"
                      onClick={() => addIntegration(credential.id)}
                    >
                      <IntegrationLogo provider={credential.provider} />
                      <span className="min-w-0 flex-1 truncate text-sm text-zinc-700">
                        {credential.title
                          ? formatCredentialName(
                              credential.title,
                              credential.provider,
                            )
                          : formatProviderName(credential.provider)}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            )}

            <div className="mt-1 border-t border-zinc-100 pt-1">
              <button
                type="button"
                disabled={isGranting}
                className="flex w-full items-center gap-2 rounded-md px-2 py-2 text-left text-sm text-zinc-700 hover:bg-zinc-50 disabled:opacity-50"
                onClick={openConnect}
              >
                <span className="flex size-6 shrink-0 items-center justify-center rounded-md bg-zinc-100 text-zinc-500">
                  <Icon icon={PlusSignIcon} size={14} />
                </span>
                Connect a new service…
              </button>
            </div>
          </PopoverContent>
        </Popover>
      </div>

      {isLoading ? (
        <Skeleton className="h-24 w-full rounded-xl" />
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
      ) : (
        <div className="divide-y divide-zinc-100 rounded-xl border border-zinc-200/80 bg-white">
          {granted.map((integration) => {
            const name = formatCredentialName(
              integration.title,
              integration.provider,
            );
            return (
              <div
                key={integration.credential_id}
                className="flex items-center gap-3 px-4 py-3"
                data-testid="expert-integration-row"
              >
                <IntegrationLogo provider={integration.provider} size={18} />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-[15px] font-medium text-zinc-800">
                    {name}
                  </div>
                  <div className="text-[13px] text-zinc-500">
                    {formatCredentialSource(integration.provider)}
                  </div>
                </div>
                <Badge variant="success" size="small">
                  Ready
                </Badge>
                <Button
                  variant="ghost"
                  size="small"
                  disabled={isRevoking}
                  aria-label={`Remove ${name}`}
                  leftIcon={<Icon icon={Delete02Icon} size={16} />}
                  onClick={() => removeIntegration(integration.credential_id)}
                >
                  Remove
                </Button>
              </div>
            );
          })}
        </div>
      )}

      <ConnectServiceDialog
        open={isConnecting}
        onOpenChange={(open) => {
          if (!open) closeConnect();
        }}
        title={
          <span className="flex items-center gap-2.5">
            <Avatar className="size-7">
              {expertAvatarUrl ? (
                <AvatarImage src={expertAvatarUrl} alt="" />
              ) : null}
              <AvatarFallback>{expertName}</AvatarFallback>
            </Avatar>
            Connect a service for {expertName}
          </span>
        }
        description={`Pick a service to connect. ${expertName} will be able to use it on your behalf.`}
        onConnected={connectCredential}
      />
    </section>
  );
}
