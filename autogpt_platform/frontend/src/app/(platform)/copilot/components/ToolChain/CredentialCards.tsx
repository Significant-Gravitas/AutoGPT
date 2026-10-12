"use client";

import { Key01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { CardProviderIcon } from "./BlockCards";
import {
  credentialDetail,
  credentialName,
  formatProviderName,
  type CredentialInventory,
} from "./credentialHelpers";
import { CARD, ChipList, RESULT_GRID } from "./ResultCards";

interface Props {
  inventory: CredentialInventory;
}

const KEY_ICON = <Icon icon={Key01Icon} size={15} className="text-zinc-600" />;

export function CredentialList({ inventory }: Props) {
  const { credentials, provisioningComplete, ungrantedProviders } = inventory;
  return (
    <div className="flex flex-col gap-1.5">
      {!provisioningComplete && (
        <p className="text-xs text-zinc-500">
          Some managed integrations could not be checked, so this list may be
          incomplete.
        </p>
      )}
      {credentials.length > 0 ? (
        <div className={RESULT_GRID}>
          {credentials.map((credential) => (
            <div
              key={credential.id}
              className={CARD + " flex items-center gap-2.5 p-2.5"}
            >
              <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
                <CardProviderIcon
                  provider={credential.provider}
                  fallback={KEY_ICON}
                />
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-[13px] font-medium text-zinc-800">
                  {credentialName(credential)}
                </p>
                <p className="truncate text-xs text-zinc-500">
                  {credentialDetail(credential)}
                </p>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-zinc-500">No connected integrations</p>
      )}
      {ungrantedProviders.length > 0 && (
        <ChipList
          label="Not granted to this expert"
          items={ungrantedProviders.map(formatProviderName)}
        />
      )}
    </div>
  );
}
