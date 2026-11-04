"use client";

import { usePutAdminSetATeamOrganizationSPlan } from "@/app/api/__generated__/endpoints/admin/admin";
import type { OrgPlanSummary } from "@/app/api/__generated__/models/orgPlanSummary";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { getErrorMessage } from "../../../../helpers";
import { refreshTeamOrgs } from "../../helpers";
import {
  getEndDateError,
  getPlanForm,
  getSeatsError,
  getSeatsWarning,
  type PlanForm,
  toPlanInput,
} from "./helpers";

export function useSetPlanDialog(org: OrgPlanSummary) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isOpen, setIsOpen] = useState(false);
  const [form, setForm] = useState(() => getPlanForm(org));
  const [error, setError] = useState<string | null>(null);
  const saveMutation = usePutAdminSetATeamOrganizationSPlan();
  const input = toPlanInput(form);

  function handleOpenChange(open: boolean) {
    setIsOpen(open);
    if (!open) return;
    setForm(getPlanForm(org));
    setError(null);
  }

  function update(patch: Partial<PlanForm>) {
    setForm((current) => ({ ...current, ...patch }));
  }

  async function save() {
    if (!input) return;
    setError(null);
    try {
      await saveMutation.mutateAsync({ orgId: org.org_id, data: input });
    } catch (saveError) {
      setError(getErrorMessage(saveError));
      return;
    }
    refreshTeamOrgs(queryClient);
    setIsOpen(false);
    toast({ title: `Saved the plan for ${org.name}`, variant: "success" });
  }

  return {
    isOpen,
    handleOpenChange,
    close: () => setIsOpen(false),
    form,
    update,
    seatsError: getSeatsError(form),
    endDateError: getEndDateError(form),
    seatsWarning: getSeatsWarning(form, org),
    error,
    canSave: input !== null,
    isSaving: saveMutation.isPending,
    save,
  };
}
