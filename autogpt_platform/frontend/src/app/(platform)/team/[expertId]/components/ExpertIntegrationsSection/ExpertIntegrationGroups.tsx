"use client";

import { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  formatCredentialName,
  groupServiceIdentity,
  serviceKey,
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

export const TYPE_LABELS: Record<string, string> = {
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
  icon: string;
  integrations: ExpertCredentialRef[];
}

export function groupExpertIntegrations(
  integrations: ExpertCredentialRef[],
): Group[] {
  const byService = new Map<string, ExpertCredentialRef[]>();
  for (const integration of integrations) {
    const id = serviceKey(integration);
    byService.set(id, [...(byService.get(id) ?? []), integration]);
  }
  return [...byService.entries()]
    .map(([id, list]) => ({
      id,
      ...groupServiceIdentity(list),
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
          // ui/accordion's content wrapper sets text-sm, which the profile tab
          // ran at 15px/1.5rem; its slot takes the same size here.
          className="overflow-hidden rounded-2xl bg-surface smooth-shadow-ring-sm [&_[data-slot=accordion-panel]]:text-body-md"
        >
          <Accordion
            type="single"
            collapsible
            defaultValue={group.id}
            className="w-full"
          >
            {/* oxlint-disable-next-line design-system/no-restyle -- AccordionItem has no borderless variant (src/components is P9's) */}
            <AccordionItem value={group.id} className="border-b-0">
              <AccordionTrigger
                textSize="body-md"
                // oxlint-disable-next-line design-system/no-restyle -- AccordionTrigger has no compact card variant (src/components is P9's)
                className="px-3 py-2.5 pr-4 hover:no-underline [&>svg]:size-4 [&>svg]:text-foreground-tertiary"
              >
                <div className="flex items-center gap-2.5">
                  <span className="inline-flex shrink-0 overflow-hidden rounded-full bg-surface">
                    <IntegrationLogo
                      provider={group.icon}
                      alt={`${group.name} logo`}
                      size={20}
                    />
                  </span>
                  <Text variant="body-md-medium" as="span" tone="primary">
                    {group.name}
                  </Text>
                  <span className="inline-flex items-center justify-center rounded-md bg-surface-hover px-1.5 py-0.5 font-sans text-caption font-medium tabular-nums text-foreground-secondary">
                    {group.integrations.length}
                  </span>
                </div>
              </AccordionTrigger>
              {/* oxlint-disable-next-line design-system/no-restyle -- AccordionContent has no flush variant (src/components is P9's) */}
              <AccordionContent className="px-0 pb-0 pt-0">
                <div className="flex flex-col divide-y divide-separator border-t border-border-base">
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
          <Text variant="body-md-medium" as="span" tone="primary">
            {name}
          </Text>
          <span className="inline-flex items-center justify-center rounded-md bg-surface-hover px-1.5 py-0.5 font-sans text-caption leading-4 text-foreground-secondary">
            {TYPE_LABELS[integration.type] ?? integration.type}
          </span>
        </div>
        <Text variant="body-md" as="span" tone="muted">
          {CONNECTION_LABELS[integration.type] ?? "Configured"}
        </Text>
      </div>
      <div className="flex items-center gap-3">
        <Text variant="eyebrow" tone="success">
          Ready
        </Text>
        <button
          type="button"
          onClick={onRemove}
          disabled={isRemoving}
          aria-busy={isRemoving}
          aria-label={`Remove ${name}`}
          className="inline-flex size-7 items-center justify-center rounded-md text-foreground-tertiary transition-colors hover:bg-surface-hover hover:text-error focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-foreground-subtle disabled:cursor-not-allowed disabled:opacity-50"
        >
          {isRemoving ? (
            <span className="inline-flex animate-spin">
              <Icon icon={Loading03Icon} size={16} />
            </span>
          ) : (
            <Icon icon={Delete02Icon} size={16} />
          )}
        </button>
      </div>
    </div>
  );
}
