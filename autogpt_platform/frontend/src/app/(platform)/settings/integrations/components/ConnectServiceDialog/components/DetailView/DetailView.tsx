"use client";

import { ArrowLeftIcon } from "@phosphor-icons/react";
import { motion, useReducedMotion } from "framer-motion";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";

import { AuthType, ConnectableProvider, type AuthMethod } from "../../helpers";
import { MethodPanel, TAB_LABEL } from "./MethodPanel";
import { ProviderAvatar } from "./ProviderAvatar";
import { UnsupportedNotice } from "./UnsupportedNotice";

interface Props {
  provider: ConnectableProvider;
  onBack: () => void;
  onSuccess: () => void;
}

const TAB_PRIORITY: AuthMethod[] = [
  AuthType.oauth2,
  AuthType.api_key,
  AuthType.user_password,
  AuthType.host_scoped,
];

// Per-provider last-selected tab. Lives in module scope so it survives dialog
// close/reopen during the same session without surviving a hard refresh.
const lastTabByProvider = new Map<string, AuthMethod>();

// Standard product-UI ease-out — keep transitions under Emil's 300ms ceiling.
const PANEL_TRANSITION = { duration: 0.18, ease: [0, 0, 0.2, 1] as const };

export function DetailView({ provider, onBack, onSuccess }: Props) {
  const description = provider.description;
  const reduceMotion = useReducedMotion();
  const tabs = TAB_PRIORITY.filter((method) =>
    provider.supportedAuthTypes.includes(method),
  );

  const remembered = lastTabByProvider.get(provider.id);
  const defaultTab =
    remembered && tabs.includes(remembered) ? remembered : tabs[0];

  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center gap-3">
        <Button
          variant="icon"
          size="icon"
          onClick={onBack}
          aria-label="Back to services"
          className="size-9"
          withTooltip={false}
        >
          <ArrowLeftIcon size={18} />
        </Button>
        <ProviderAvatar id={provider.id} name={provider.name} />
        <div className="flex min-w-0 flex-col gap-1">
          <Text variant="h4" as="h2" className="text-[#1F1F20]">
            {provider.name}
          </Text>
          {description ? (
            <Text variant="small" className="truncate text-[#83838C]">
              {description}
            </Text>
          ) : null}
        </div>
      </div>

      {tabs.length === 0 ? (
        <UnsupportedNotice providerName={provider.name} />
      ) : tabs.length === 1 ? (
        <MethodPanel
          method={tabs[0]}
          provider={provider}
          onSuccess={onSuccess}
        />
      ) : (
        <TabsLine
          defaultValue={defaultTab}
          onValueChange={(value) => {
            const next = value as AuthMethod;
            if (tabs.includes(next)) lastTabByProvider.set(provider.id, next);
          }}
        >
          <TabsLineList>
            {tabs.map((method) => (
              <TabsLineTrigger key={method} value={method}>
                {TAB_LABEL[method]}
              </TabsLineTrigger>
            ))}
          </TabsLineList>
          {tabs.map((method) => (
            <TabsLineContent key={method} value={method}>
              <motion.div
                initial={
                  reduceMotion
                    ? { opacity: 0 }
                    : { opacity: 0, x: 6, filter: "blur(2px)" }
                }
                animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
                transition={PANEL_TRANSITION}
              >
                <MethodPanel
                  method={method}
                  provider={provider}
                  onSuccess={onSuccess}
                />
              </motion.div>
            </TabsLineContent>
          ))}
        </TabsLine>
      )}
    </div>
  );
}
