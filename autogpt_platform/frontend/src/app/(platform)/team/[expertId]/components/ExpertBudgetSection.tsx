"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import {
  BUTTON_ICON_SIZE,
  extendedButtonVariants,
} from "@/components/atoms/Button/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { creditsToUsdLabel } from "@/lib/credits";
import { cn } from "@/lib/utils";
import {
  InformationCircleIcon,
  PencilEdit02Icon,
} from "@hugeicons/core-free-icons";
import { useState } from "react";
import { SpendMeter } from "../../components/ExpertTeamCard/components/SpendMeter";
import { getWeeklySpend } from "../../helpers";
import { EditBudgetDialog } from "./EditBudgetDialog/EditBudgetDialog";
import { isOnSubscription } from "./ExpertLlmRouteSection/helpers";

interface Props {
  expert: Expert;
  /** The cover colour it sits on: text and meter take darker shades of it. */
  color?: string;
  className?: string;
}

export function ExpertBudgetSection({ expert, color, className }: Props) {
  const weeklySpend = getWeeklySpend(expert);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const isChatUnmetered =
    isOnSubscription(expert) && expert.llm_route_available !== false;

  return (
    <section
      aria-label={`${expert.name} budget`}
      style={
        color
          ? { color: `color-mix(in srgb, ${color} 55%, #18181b)` }
          : undefined
      }
      className={cn("flex w-full shrink-0 flex-col gap-1.5", className)}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1">
          <Text
            variant="body-medium"
            tone="secondary"
            className={cn(color && "!text-inherit")}
          >
            Budget
          </Text>
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            leadingIcon={PencilEdit02Icon}
            aria-label="Edit budget"
            className={cn(color && "text-inherit hover:bg-white/50")}
            onClick={() => setIsEditOpen(true)}
          />
          {isChatUnmetered ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-label="Budget metering"
                  className={cn(
                    extendedButtonVariants({
                      variant: "ghost",
                      size: "icon-xs",
                    }),
                    "shrink-0",
                  )}
                >
                  <Icon
                    icon={InformationCircleIcon}
                    size={BUTTON_ICON_SIZE["icon-xs"]}
                    aria-hidden
                  />
                </button>
              </TooltipTrigger>
              <TooltipContent>
                Chat is not metered on {expert.llm_route_label}. Only workflow
                runs on platform credits count here.
              </TooltipContent>
            </Tooltip>
          ) : null}
        </div>
        <Text
          variant="body-medium"
          tone="secondary"
          unmask={false}
          className={cn("tabular-nums", color && "!text-inherit")}
        >
          {weeklySpend
            ? `${creditsToUsdLabel(weeklySpend.spent)} / ${creditsToUsdLabel(weeklySpend.budget)}`
            : "No budget"}
        </Text>
      </div>
      <SpendMeter
        spent={weeklySpend?.spent ?? 0}
        budget={weeklySpend?.budget ?? 1}
        muted={!weeklySpend}
        color={color}
      />
      <EditBudgetDialog
        expert={expert}
        open={isEditOpen}
        onClose={() => setIsEditOpen(false)}
      />
    </section>
  );
}
