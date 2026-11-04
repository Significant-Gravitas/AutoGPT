"use client";

import { usePostAdminAddCreditsToATeamOrganization } from "@/app/api/__generated__/endpoints/admin/admin";
import type { OrgCreditGrantResult } from "@/app/api/__generated__/models/orgCreditGrantResult";
import type { OrgPlanSummary } from "@/app/api/__generated__/models/orgPlanSummary";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { getErrorMessage } from "../../../../helpers";
import { formatBalance, formatCredits, refreshTeamOrgs } from "../../helpers";
import { describeRepeat, getAmountError, toCreditsInput } from "./helpers";

export function useAddCreditsDialog(org: OrgPlanSummary) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isOpen, setIsOpen] = useState(false);
  const [amount, setAmount] = useState("");
  const [reference, setReference] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const addMutation = usePostAdminAddCreditsToATeamOrganization();
  const input = toCreditsInput(amount, reference);

  function handleOpenChange(open: boolean) {
    setIsOpen(open);
    if (!open) return;
    setAmount("");
    setReference("");
    setNotice(null);
    setError(null);
  }

  function changeAmount(value: string) {
    setAmount(value);
    setNotice(null);
  }

  function changeReference(value: string) {
    setReference(value);
    setNotice(null);
  }

  async function add() {
    if (!input) return;
    setError(null);
    setNotice(null);
    let grant: OrgCreditGrantResult | undefined;
    try {
      grant = okData(
        await addMutation.mutateAsync({ orgId: org.org_id, data: input }),
      );
    } catch (addError) {
      setError(getErrorMessage(addError));
      return;
    }
    refreshTeamOrgs(queryClient);
    if (grant && !grant.granted) {
      setNotice(describeRepeat(grant));
      return;
    }
    setIsOpen(false);
    toast({
      title: `Added ${formatCredits(input.amount)} to ${org.name}`,
      description: grant
        ? `The balance is now ${formatBalance(grant.balance)}.`
        : undefined,
      variant: "success",
    });
  }

  return {
    isOpen,
    handleOpenChange,
    close: () => setIsOpen(false),
    amount,
    changeAmount,
    amountError: getAmountError(amount),
    reference,
    changeReference,
    notice,
    error,
    canAdd: input !== null,
    isAdding: addMutation.isPending,
    add,
  };
}
