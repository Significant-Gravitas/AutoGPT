"use client";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { ConnectProviderRow } from "@/app/(platform)/copilot/components/OnboardingWelcomeDialog/ConnectProviderRow";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  formatCredentialName,
  formatProviderName,
} from "@/components/contextual/IntegrationsPanel/helpers";
import type { ConnectableProvider } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/helpers";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { useBottomScrollShadow } from "../../../components/SoulDrawer/useBottomScrollShadow";
import { useFitListToDialog } from "../useFitListToDialog";

interface Props {
  open: boolean;
  expertName: string;
  credentials: CredentialsMetaResponse[];
  isGranting: boolean;
  onUse: (credentialId: string) => void;
  onClose: () => void;
}

interface ProviderGroup {
  provider: ConnectableProvider;
  credentials: CredentialsMetaResponse[];
}

function groupByProvider(
  credentials: CredentialsMetaResponse[],
): ProviderGroup[] {
  const groups = new Map<string, CredentialsMetaResponse[]>();
  for (const credential of credentials) {
    groups.set(credential.provider, [
      ...(groups.get(credential.provider) ?? []),
      credential,
    ]);
  }
  return [...groups.entries()]
    .map(([id, list]) => ({
      provider: { id, name: formatProviderName(id), supportedAuthTypes: [] },
      credentials: list,
    }))
    .sort((a, b) => a.provider.name.localeCompare(b.provider.name));
}

/** Credentials already on your account that this expert cannot use yet,
 *  grouped like the connect picker. A provider with one credential grants
 *  on click; with several, a second step asks which one. */
export function UseExistingCredentialsDialog({
  open,
  expertName,
  credentials,
  isGranting,
  onUse,
  onClose,
}: Props) {
  const [query, setQuery] = useState("");
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const { attachList, list } = useFitListToDialog<HTMLUListElement>();
  const hasMoreBelow = useBottomScrollShadow(list);

  useEffect(() => {
    if (!open) {
      setQuery("");
      setSelectedProvider(null);
    }
  }, [open]);

  const needle = query.trim().toLowerCase();
  const groups = groupByProvider(credentials).filter(
    (group) =>
      !needle ||
      group.provider.name.toLowerCase().includes(needle) ||
      group.credentials.some((credential) =>
        (credential.title ?? "").toLowerCase().includes(needle),
      ),
  );
  const selected = groups.find(
    (group) => group.provider.id === selectedProvider,
  );

  function handleSelect(providerId: string) {
    const group = groups.find((item) => item.provider.id === providerId);
    if (!group) return;
    if (group.credentials.length === 1) {
      onUse(group.credentials[0].id);
      return;
    }
    setSelectedProvider(providerId);
  }

  return (
    <Dialog
      variant="compact"
      styling={{ maxWidth: "40rem", maxHeight: "60vh" }}
      title="Use an existing connection"
      controlled={{
        isOpen: open,
        set: (next) => {
          if (!next) onClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="-mb-2 flex flex-col gap-3">
          {selected ? (
            <>
              <button
                type="button"
                onClick={() => setSelectedProvider(null)}
                className="flex w-fit items-center gap-1 text-sm text-zinc-500 hover:text-zinc-800"
              >
                <Icon icon={ArrowLeft02Icon} size={14} />
                All connections
              </button>
              <Text variant="body" tone="muted" className="leading-5">
                You have {selected.credentials.length} {selected.provider.name}{" "}
                connections. Choose which one {expertName} should use.
              </Text>
              <ul
                className="flex flex-col gap-2"
                aria-label={`${selected.provider.name} connections`}
              >
                {selected.credentials.map((credential) => {
                  const name = formatCredentialName(
                    credential.title ?? credential.provider,
                    credential.provider,
                  );
                  return (
                    <li
                      key={credential.id}
                      className="flex items-center gap-3 rounded-xl bg-neutral-100 px-3 py-2.5"
                    >
                      <IntegrationLogo
                        provider={credential.provider}
                        size={24}
                      />
                      <Text
                        variant="body-medium"
                        as="span"
                        tone="primary"
                        className="min-w-0 flex-1 truncate"
                      >
                        {name}
                      </Text>
                      <Button
                        variant="secondary"
                        size="xs"
                        disabled={isGranting}
                        onClick={() => onUse(credential.id)}
                        aria-label={`Let ${expertName} use ${name}`}
                      >
                        Use
                      </Button>
                    </li>
                  );
                })}
              </ul>
            </>
          ) : (
            <>
              <Text variant="body" tone="muted" className="leading-5">
                These connections are already set up on your account but{" "}
                {expertName} can&apos;t use them yet. Choose one to give{" "}
                {expertName} access. Nothing is re-authorised, and you can
                remove the access from this tab at any time.
              </Text>
              <SearchInput
                size="small"
                value={query}
                onChange={setQuery}
                placeholder="Search connections"
              />
              {groups.length === 0 ? (
                <Text variant="small" tone="muted">
                  {credentials.length === 0
                    ? `${expertName} can already use everything in your account.`
                    : "No connections match."}
                </Text>
              ) : (
                <div className="relative">
                  <ul
                    ref={attachList}
                    className="grid grid-cols-2 gap-2 overflow-y-auto pr-1"
                    aria-label="Existing connections"
                  >
                    {groups.map((group) => (
                      <li key={group.provider.id}>
                        <ConnectProviderRow
                          provider={group.provider}
                          onSelect={handleSelect}
                          description={
                            group.credentials.length === 1
                              ? formatCredentialName(
                                  group.credentials[0].title ??
                                    group.provider.name,
                                  group.provider.id,
                                )
                              : `${group.credentials.length} connections · choose one`
                          }
                        />
                      </li>
                    ))}
                  </ul>

                  <div
                    aria-hidden="true"
                    className={cn(
                      "pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-white to-transparent transition-opacity duration-200",

                      hasMoreBelow ? "opacity-100" : "opacity-0",
                    )}
                  />
                </div>
              )}
            </>
          )}
          <div className="flex justify-end pt-1">
            <Button variant="secondary" size="xs" onClick={onClose}>
              Cancel
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
