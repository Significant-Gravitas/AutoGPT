"use client";

import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ProviderAvatar } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/ProviderAvatar";
import type { ApiKeyConnectFormValues } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/schema";
import { UnsupportedNotice } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/DetailView/UnsupportedNotice";
import {
  AuthType,
  type AuthMethod,
  type ConnectableProvider,
} from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/helpers";
import type { UseFormReturn } from "react-hook-form";
import { InlineApiKeyForm } from "./InlineApiKeyForm";
import {
  GlobeIcon,
  Key01Icon,
  SecurityCheckIcon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { AnimatePresence, motion } from "framer-motion";

interface Props {
  provider: ConnectableProvider;
  selectedMethod: AuthMethod | null;
  onSelectMethod: (method: AuthMethod) => void;
  apiKeyForm: UseFormReturn<ApiKeyConnectFormValues>;
  onApiKeySubmit: (values: ApiKeyConnectFormValues) => void;
}

const METHOD_ORDER: AuthMethod[] = [
  AuthType.oauth2,
  AuthType.api_key,
  AuthType.user_password,
  AuthType.host_scoped,
];

const METHOD_COPY: Record<
  AuthMethod,
  {
    label: string;
    description: string;
    icon: IconSvgElement;
    recommended?: boolean;
  }
> = {
  [AuthType.oauth2]: {
    label: "OAuth",
    description: "One-click secure access.",
    icon: SecurityCheckIcon,
    recommended: true,
  },
  [AuthType.api_key]: {
    label: "API Key",
    description: "Paste your token for a custom setup.",
    icon: Key01Icon,
  },
  [AuthType.user_password]: {
    label: "Username & password",
    description: "Sign in with account credentials.",
    icon: UserIcon,
  },
  [AuthType.host_scoped]: {
    label: "Host",
    description: "Scope credentials to one host.",
    icon: GlobeIcon,
  },
};

// The connect step behind a provider click: logo pair, "Connect AutoGPT
// to X", method radio cards. Form methods expand their inputs inline as
// an accordion; OAuth is driven by the panel footer's Continue.
export function ConnectMethodView({
  provider,
  selectedMethod,
  onSelectMethod,
  apiKeyForm,
  onApiKeySubmit,
}: Props) {
  const methods = METHOD_ORDER.filter((method) =>
    provider.supportedAuthTypes.includes(method),
  );

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
          <ProviderAvatar id={provider.id} name={provider.name} />
        </span>
      </div>

      <div className="flex flex-col gap-1.5 text-center">
        <Text variant="h3" className="!text-[1.25rem] text-zinc-900">
          Connect AutoGPT to {provider.name}
        </Text>
        <Text variant="body" className="!text-zinc-500">
          Choose how you&apos;d like to connect your {provider.name} account.
        </Text>
      </div>

      <div className="flex flex-col gap-1 rounded-2xl bg-neutral-100 p-1.5">
        {methods.map((method) => {
          const copy = METHOD_COPY[method];
          const isSelected = selectedMethod === method;
          const hasInlineForm = method !== AuthType.oauth2;
          return (
            <div
              key={method}
              className={isSelected ? "rounded-xl bg-white shadow-sm" : ""}
            >
              <button
                type="button"
                onClick={() => onSelectMethod(method)}
                className={
                  isSelected
                    ? "flex w-full items-center gap-3 rounded-xl p-3 text-left"
                    : "flex w-full items-center gap-3 rounded-xl p-3 text-left transition-colors hover:bg-white/60"
                }
              >
                <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-lg bg-white shadow-sm">
                  <Icon icon={copy.icon} size={22} className="text-zinc-700" />
                </span>
                <span className="flex min-w-0 flex-1 flex-col gap-0.5">
                  <span className="flex items-center gap-2">
                    <span className="text-sm font-semibold text-zinc-900">
                      {copy.label}
                    </span>
                    {copy.recommended && (
                      <span className="rounded-md bg-violet-100 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-violet-700">
                        RECOMMENDED
                      </span>
                    )}
                  </span>
                  <span className="text-xs text-zinc-500">
                    {copy.description}
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
              {/* Form methods open in place: the card grows to reveal the
                  inputs, submit included — no separate step. */}
              <AnimatePresence initial={false}>
                {isSelected && hasInlineForm && (
                  <motion.div
                    key="form"
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.25, ease: [0, 0, 0.2, 1] }}
                    className="overflow-hidden"
                  >
                    <div className="px-3 pb-3">
                      {method === AuthType.api_key ? (
                        <InlineApiKeyForm
                          form={apiKeyForm}
                          providerName={provider.name}
                          onSubmit={onApiKeySubmit}
                        />
                      ) : (
                        <UnsupportedNotice providerName={provider.name} />
                      )}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          );
        })}
      </div>
    </div>
  );
}
