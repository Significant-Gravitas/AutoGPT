"use client";

import { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  formatCredentialName,
  formatCredentialSource,
  formatProviderName,
} from "@/components/contextual/IntegrationsPanel/helpers";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/molecules/Accordion/Accordion";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { Delete02Icon, Loading03Icon } from "@hugeicons/core-free-icons";
import { motion, useReducedMotion, type Variants } from "framer-motion";

const TYPE_LABELS: Record<string, string> = {
  api_key: "API Key",
  oauth2: "OAuth",
  user_password: "Username & password",
  host_scoped: "Host",
  device_code: "Device auth",
};

const CONNECTION_LABELS: Record<string, string> = {
  api_key: "API key configured",
  oauth2: "Connected via OAuth",
  user_password: "Username/password set",
  host_scoped: "Host configured",
  device_code: "Connected via device auth",
};

function formatConnection(integration: ExpertCredentialRef): string {
  if (integration.provider === "mcp") return formatCredentialSource("mcp");
  return CONNECTION_LABELS[integration.type] ?? "Configured";
}

const CONTAINER_VARIANTS: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.08, delayChildren: 0.05 } },
};
const ITEM_VARIANTS: Variants = {
  hidden: { opacity: 0, y: 16 },
  show: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.3, ease: [0.16, 1, 0.3, 1] },
  },
};
const REDUCED_ITEM_VARIANTS: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1 },
};

interface Group {
  id: string;
  name: string;
  integrations: ExpertCredentialRef[];
}

export function groupExpertIntegrations(
  integrations: ExpertCredentialRef[],
): Group[] {
  const byProvider = new Map<string, ExpertCredentialRef[]>();
  for (const integration of integrations) {
    const id =
      integration.provider === "codex" ? "openai" : integration.provider;
    byProvider.set(id, [...(byProvider.get(id) ?? []), integration]);
  }
  return [...byProvider.entries()]
    .map(([id, list]) => ({
      id,
      name: formatProviderName(id),
      integrations: list,
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

interface Props {
  integrations: ExpertCredentialRef[];
  isRemoving: boolean;
  onRemove: (credentialId: string) => void;
}

export function ExpertIntegrationGroups({
  integrations,
  isRemoving,
  onRemove,
}: Props) {
  const reduceMotion = useReducedMotion();
  const groups = groupExpertIntegrations(integrations);

  return (
    <motion.div
      className="flex flex-col gap-3"
      variants={CONTAINER_VARIANTS}
      initial="hidden"
      animate="show"
    >
      {groups.map((group) => (
        <motion.div
          key={group.id}
          variants={reduceMotion ? REDUCED_ITEM_VARIANTS : ITEM_VARIANTS}
        >
          <Accordion
            type="single"
            collapsible
            defaultValue={group.id}
            className="w-full overflow-hidden rounded-xl border border-[#DADADC] bg-white"
          >
            <AccordionItem value={group.id} className="border-b-0">
              <AccordionTrigger className="px-3 py-3 pr-5 hover:no-underline [&>svg]:text-[#1F1F20]">
                <div className="flex items-center gap-3">
                  <IntegrationLogo
                    provider={group.id}
                    alt={`${group.name} logo`}
                    size={24}
                    className="rounded-full bg-white"
                  />
                  <span className="text-[16px] font-medium leading-[26px] tracking-[-0.08px] text-black">
                    {group.name}
                  </span>
                  <span className="inline-flex items-center justify-center rounded-md bg-[#EFF1F4] px-2 py-[2px] text-[14px] font-medium leading-[22px] text-black">
                    {group.integrations.length}
                  </span>
                </div>
              </AccordionTrigger>
              <AccordionContent className="px-0 pb-0 pt-0">
                <div className="flex flex-col divide-y divide-[#DADADC] border-t border-[#DADADC]">
                  {group.integrations.map((integration) => (
                    <ExpertIntegrationRow
                      key={integration.credential_id}
                      integration={integration}
                      isRemoving={isRemoving}
                      onRemove={() => onRemove(integration.credential_id)}
                    />
                  ))}
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </motion.div>
      ))}
    </motion.div>
  );
}

interface RowProps {
  integration: ExpertCredentialRef;
  isRemoving: boolean;
  onRemove: () => void;
}

function ExpertIntegrationRow({ integration, isRemoving, onRemove }: RowProps) {
  const name = formatCredentialName(integration.title, integration.provider);
  return (
    <div
      className="flex w-full items-center justify-between py-3 pl-4 pr-5"
      data-testid="expert-integration-row"
    >
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-3">
          <span className="text-[14px] font-medium leading-[22px] text-[#1F1F20]">
            {name}
          </span>
          <span className="inline-flex items-center justify-center rounded-md bg-[#EFF1F4] px-2 py-[2px] text-[12px] font-medium leading-[20px] text-[#505057]">
            {TYPE_LABELS[integration.type] ?? integration.type}
          </span>
        </div>
        <span className="text-[11px] font-medium uppercase tracking-[1.1px] text-[#505057]">
          {formatConnection(integration)}
        </span>
      </div>
      <div className="flex items-center gap-4">
        <span className="text-[11px] font-medium uppercase tracking-[1.1px] text-emerald-600">
          Ready
        </span>
        <button
          type="button"
          onClick={onRemove}
          disabled={isRemoving}
          aria-busy={isRemoving}
          aria-label={`Remove ${name}`}
          className="inline-flex size-5 items-center justify-center text-[#1F1F20] transition-colors hover:text-red-500 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-purple-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isRemoving ? (
            <Icon icon={Loading03Icon} size={20} className="animate-spin" />
          ) : (
            <Icon icon={Delete02Icon} size={20} />
          )}
        </button>
      </div>
    </div>
  );
}
