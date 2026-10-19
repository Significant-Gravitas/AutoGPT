"use client";

import { PlugSocketIcon } from "@hugeicons/core-free-icons";
import { groupExpertIntegrations } from "@/app/(platform)/team/[expertId]/components/ExpertIntegrationsSection/ExpertIntegrationGroups";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Icon } from "@/components/atoms/Icon/Icon";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { cn } from "@/lib/utils";
import { useCopilotUIStore, type ContextPanelExpert } from "../../../../store";
import { useExpertIntegrations } from "./useExpertIntegrations";

const VISIBLE_LOGOS = 2;

interface Props {
  expert: ContextPanelExpert;
  className?: string;
  /** `lg` matches the mobile thread chip's buttons. */
  size?: "md" | "lg";
}

/** The services the chat's expert can reach, as a stack of logos: the first
 *  two name themselves on hover and "+N" names the rest. With none yet, a
 *  plug icon stands in so the panel's add actions stay one click away.
 *  Clicking opens the integrations tab of the side panel. */
export function IntegrationsToggle({ expert, className, size = "md" }: Props) {
  const isLarge = size === "lg";
  const { integrations } = useExpertIntegrations(expert.id);
  const toggleIntegrationsPanel = useCopilotUIStore(
    (s) => s.toggleIntegrationsPanel,
  );
  const isActive = useCopilotUIStore(
    (s) =>
      s.artifactPanel.isOpen &&
      s.artifactPanel.activeTab === "integrations" &&
      s.artifactPanel.activeArtifact == null &&
      !s.artifactPanel.isComputerOpen,
  );

  const services = groupExpertIntegrations(integrations);

  if (services.length === 0) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={() => toggleIntegrationsPanel(expert)}
            aria-label={
              isActive ? "Hide integrations" : `${expert.name}'s integrations`
            }
            aria-pressed={isActive}
            data-testid="expert-integrations-empty"
            className={cn(
              className,
              "flex items-center justify-center",
              isLarge ? "size-9" : "size-8",
              isActive && "bg-zinc-100",
            )}
          >
            <Icon
              icon={PlugSocketIcon}
              className={cn(
                "text-sidebar-foreground/90",
                isLarge ? "!size-5" : "!size-4",
              )}
            />
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          {isActive ? "Hide integrations" : "Integrations"}
        </TooltipContent>
      </Tooltip>
    );
  }

  const shown = services.slice(0, VISIBLE_LOGOS);
  const hidden = services.slice(VISIBLE_LOGOS);
  const hiddenNames = hidden.map((service) => service.name).join(", ");
  const allNames = services.map((service) => service.name).join(", ");

  return (
    <TooltipProvider>
      <button
        type="button"
        onClick={() => toggleIntegrationsPanel(expert)}
        aria-label={
          isActive
            ? "Hide integrations"
            : `${expert.name}'s integrations: ${allNames}`
        }
        aria-pressed={isActive}
        data-testid="expert-integrations"
        className={cn(
          className,
          "flex items-center gap-1.5 px-1.5",
          isLarge ? "h-9 w-auto" : "h-8",
          isActive && "bg-zinc-100",
        )}
      >
        <span className="flex -space-x-1.5">
          {shown.map((service) => (
            <Tooltip key={service.id}>
              <TooltipTrigger asChild>
                <span
                  className={cn(
                    "flex items-center justify-center rounded-full bg-white ring-2 ring-white smooth-shadow-ring-sm",
                    isLarge ? "size-7" : "size-6",
                  )}
                >
                  <IntegrationLogo
                    provider={service.icon}
                    alt={service.name}
                    size={isLarge ? 18 : 14}
                  />
                </span>
              </TooltipTrigger>
              <TooltipContent side="bottom">{service.name}</TooltipContent>
            </Tooltip>
          ))}
        </span>
        {hidden.length > 0 && (
          <Tooltip>
            <TooltipTrigger asChild>
              <span className="text-xs font-medium tabular-nums text-sidebar-foreground/90">
                +{hidden.length}
              </span>
            </TooltipTrigger>
            <TooltipContent side="bottom">{hiddenNames}</TooltipContent>
          </Tooltip>
        )}
      </button>
    </TooltipProvider>
  );
}
