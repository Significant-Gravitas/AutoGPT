import {
  getGetExpertQueryKey,
  getListExpertsQueryKey,
  useUpdateExpertBudget,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { toast } from "@/components/molecules/Toast/use-toast";
import { CREDITS_PER_USD, parseUsdToCredits } from "@/lib/credits";
import { useQueryClient } from "@tanstack/react-query";
import { FormEvent, useEffect, useState } from "react";

export const WEEKLY_BUDGET_MAX_CREDITS = 1_000_000;

interface Args {
  expert: Expert;
  open: boolean;
  onClose: () => void;
}

function toUsdInput(credits: number | null | undefined) {
  if (credits == null) return "";
  const dollars = credits / CREDITS_PER_USD;
  return Number.isInteger(dollars) ? String(dollars) : dollars.toFixed(2);
}

export function useEditBudgetDialog({ expert, open, onClose }: Args) {
  const queryClient = useQueryClient();
  const [value, setValue] = useState(() => toUsdInput(expert.weekly_budget));

  useEffect(() => {
    if (open) setValue(toUsdInput(expert.weekly_budget));
  }, [open, expert.weekly_budget]);

  const trimmed = value.trim();
  const credits = trimmed
    ? parseUsdToCredits(trimmed, WEEKLY_BUDGET_MAX_CREDITS)
    : null;
  const isInvalid = trimmed.length > 0 && credits === null;

  const { mutate, isPending } = useUpdateExpertBudget({
    mutation: {
      onSuccess: () => {
        queryClient.invalidateQueries({
          queryKey: getGetExpertQueryKey(expert.id),
        });
        queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() });
        toast({ title: "Budget updated" });
        onClose();
      },
      onError: () => {
        toast({
          title: "Could not update the budget",
          description: "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  function save(event: FormEvent) {
    event.preventDefault();
    if (isInvalid || isPending) return;
    mutate({ expertId: expert.id, data: { weekly_budget: credits } });
  }

  return { value, setValue, isInvalid, isPending, save };
}
