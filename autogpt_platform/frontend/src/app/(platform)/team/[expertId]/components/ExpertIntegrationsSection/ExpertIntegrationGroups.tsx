"use client";

import { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
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
      className="flex flex-col gap-2.5"
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
            className="w-full overflow-hidden rounded-lg border border-zinc-200 bg-white"
          >
            <AccordionItem value={group.id} className="border-b-0">
              <AccordionTrigger className="px-3 py-2.5 pr-4 hover:no-underline [&>svg]:size-4 [&>svg]:text-zinc-500">
                <div className="flex items-center gap-2.5">
                  <IntegrationLogo
                    provider={group.id}
                    alt={`${group.name} logo`}
                    size={20}
                    className="rounded-full bg-white"
                  />
                  <Text variant="body-medium" as="span" tone="primary">
                    {group.name}
                  </Text>
                  <Text
                    variant="small-medium"
                    as="span"
                    tone="secondary"
                    className="inline-flex items-center justify-center rounded-md bg-zinc-100 px-1.5 py-0.5 tabular-nums"
                  >
                    {group.integrations.length}
                  </Text>
                </div>
              </AccordionTrigger>
              <AccordionContent className="px-0 pb-0 pt-0">
                <div className="flex flex-col divide-y divide-zinc-100 border-t border-zinc-200">
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
      className="flex w-full items-center justify-between py-2.5 pl-3 pr-4"
      data-testid="expert-integration-row"
    >
      <div className="flex flex-col gap-0.5">
        <div className="flex items-center gap-2">
          <Text variant="body-medium" as="span" tone="primary">
            {name}
          </Text>
          <Text
            variant="small"
            as="span"
            tone="secondary"
            className="inline-flex items-center justify-center rounded-md bg-zinc-100 px-1.5 py-0.5 leading-4"
          >
            {TYPE_LABELS[integration.type] ?? integration.type}
          </Text>
        </div>
        <Text variant="body" as="span" tone="muted">
          {formatConnection(integration)}
        </Text>
      </div>
      <div className="flex items-center gap-3">
        <Text variant="eyebrow" className="text-emerald-600">
          Ready
        </Text>
        <button
          type="button"
          onClick={onRemove}
          disabled={isRemoving}
          aria-busy={isRemoving}
          aria-label={`Remove ${name}`}
          className="inline-flex size-7 items-center justify-center rounded-md text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-red-500 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isRemoving ? (
            <Icon icon={Loading03Icon} size={16} className="animate-spin" />
          ) : (
            <Icon icon={Delete02Icon} size={16} />
          )}
        </button>
      </div>
    </div>
  );
}
