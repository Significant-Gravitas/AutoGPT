"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ProviderAvatar } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/ProviderAvatar";
import { Key01Icon, SecurityCheckIcon } from "@hugeicons/core-free-icons";
import type { ExistingCredential } from "../../helpers";

interface Props {
  provider: string;
  displayName: string;
  credentials: ExistingCredential[];
  selectedId: string;
  onSelect: (id: string) => void;
}

// The step before the connect methods when the account already has a
// credential the expert lacks: pick which one to hand over. Mirrors the
// method radio cards so switching to Add new feels like the same dialog.
export function ExistingCredentialsView({
  provider,
  displayName,
  credentials,
  selectedId,
  onSelect,
}: Props) {
  return (
    <div className="flex flex-col gap-5 pt-2">
      <div className="flex items-center justify-center gap-6">
        <span className="relative flex h-20 w-20 items-center justify-center rounded-full bg-white shadow-[0_8px_24px_rgba(0,0,0,0.08)] ring-1 ring-zinc-100">
          <AutoGPTLogo
            hideText
            className="absolute left-1/2 top-1/2 h-8 w-[4.4rem] -translate-x-[77%] -translate-y-1/2"
          />
        </span>
        <span aria-hidden className="grid grid-cols-3 gap-1.5">
          {Array.from({ length: 9 }, (_, dot) => (
            <span key={dot} className="h-1 w-1 rounded-full bg-[#5b21b6]/30" />
          ))}
        </span>
        <span className="flex h-20 w-20 items-center justify-center rounded-full bg-white shadow-[0_8px_24px_rgba(0,0,0,0.08)] ring-1 ring-zinc-100">
          <ProviderAvatar id={provider} name={displayName} />
        </span>
      </div>

      <div className="flex flex-col gap-1.5 text-center">
        <Text variant="h3" className="!text-[1.25rem] text-zinc-900">
          Give this expert access to {displayName}
        </Text>
        <Text variant="body" className="!text-zinc-500">
          Your account is already connected, but this expert can&apos;t use it
          yet. Pick an account to share, or add a new one.
        </Text>
      </div>

      <div
        role="radiogroup"
        aria-label="Account"
        className="flex flex-col gap-1 rounded-2xl bg-neutral-100 p-1.5"
      >
        {credentials.map((credential) => {
          const isSelected = credential.id === selectedId;
          return (
            <button
              key={credential.id}
              type="button"
              role="radio"
              aria-checked={isSelected}
              onClick={() => onSelect(credential.id)}
              className={
                isSelected
                  ? "flex w-full items-center gap-3 rounded-xl bg-white p-3 text-left shadow-sm"
                  : "flex w-full items-center gap-3 rounded-xl p-3 text-left transition-colors hover:bg-white/60"
              }
            >
              <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-white shadow-sm">
                <Icon
                  icon={
                    credential.type === "oauth2" ? SecurityCheckIcon : Key01Icon
                  }
                  size={22}
                  className="text-zinc-700"
                />
              </span>
              <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                <span className="truncate text-sm font-semibold text-zinc-900">
                  {credential.title}
                </span>
                <span className="text-xs text-zinc-500">
                  {credential.type === "oauth2" ? "OAuth" : "API key"}
                </span>
              </span>
              <span
                aria-hidden
                className={
                  isSelected
                    ? "flex h-5 w-5 shrink-0 items-center justify-center rounded-full border-2 border-violet-600"
                    : "h-5 w-5 shrink-0 rounded-full border-2 border-zinc-200"
                }
              >
                {isSelected && (
                  <span className="h-2.5 w-2.5 rounded-full bg-violet-600" />
                )}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
