"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { creditsToUsdLabel } from "@/lib/credits";
import { PencilEdit02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { SpendMeter } from "../../components/ExpertTeamCard/components/SpendMeter";
import { getWeeklySpend } from "../../helpers";
import { EditBudgetDialog } from "./EditBudgetDialog/EditBudgetDialog";

interface Props {
  expert: Expert;
}

export function ExpertBudgetSection({ expert }: Props) {
  const weeklySpend = getWeeklySpend(expert);
  const [isEditOpen, setIsEditOpen] = useState(false);

  return (
    <section
      aria-label={`${expert.name} budget`}
      className="flex w-full shrink-0 flex-col gap-1.5 md:ml-auto md:w-64"
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1">
          <Text variant="body-medium" tone="secondary">
            Budget
          </Text>
          <Button
            type="button"
            variant="ghost"
            size="icon-xs"
            leadingIcon={PencilEdit02Icon}
            aria-label="Edit budget"
            onClick={() => setIsEditOpen(true)}
          />
        </div>
        <Text
          variant="body-medium"
          tone="secondary"
          unmask={false}
          className="tabular-nums"
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
      />
      <EditBudgetDialog
        expert={expert}
        open={isEditOpen}
        onClose={() => setIsEditOpen(false)}
      />
    </section>
  );
}
