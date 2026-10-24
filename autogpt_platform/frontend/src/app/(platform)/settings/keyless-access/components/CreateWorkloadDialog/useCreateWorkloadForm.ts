"use client";

import { useState } from "react";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";

import type { WorkloadIdentityInfo } from "@/app/api/__generated__/models/workloadIdentityInfo";
import { environment } from "@/services/environment";

import {
  buildWorkloadSpec,
  EMPTY_FORM_VALUES,
  type Platform,
  type WorkloadFormValues,
} from "../../helpers";
import { useWorkloadMutations } from "../hooks/useWorkloadMutations";
import { workloadFormSchema } from "./schema";

type View = "platform" | "details" | "success";

interface Args {
  onClose: () => void;
}

export function useCreateWorkloadForm({ onClose }: Args) {
  const [view, setView] = useState<View>("platform");
  const [created, setCreated] = useState<WorkloadIdentityInfo | null>(null);
  const { createIdentity, isCreating } = useWorkloadMutations();

  const form = useForm<WorkloadFormValues>({
    resolver: zodResolver(workloadFormSchema),
    defaultValues: EMPTY_FORM_VALUES,
    mode: "onChange",
  });
  const values = form.watch();
  const platform = values.platform;
  // formState.isValid is never recomputed for a form whose fields mount after
  // the platform step, so validity comes from the schema itself.
  const canSubmit = workloadFormSchema.safeParse(values).success;

  function choosePlatform(next: Platform) {
    form.setValue("platform", next, { shouldValidate: true });
    setView("details");
  }

  async function handleSubmit(values: WorkloadFormValues) {
    const audience = environment.getAGPTServerBaseUrl();
    try {
      setCreated(await createIdentity(buildWorkloadSpec(values, audience)));
      setView("success");
    } catch {
      // The mutation's toast explains the failure; keep the form to retry.
    }
  }

  function handleBack() {
    setView("platform");
  }

  function handleClose() {
    setView("platform");
    setCreated(null);
    form.reset(EMPTY_FORM_VALUES);
    onClose();
  }

  return {
    form,
    view,
    platform,
    canSubmit,
    created,
    isCreating,
    choosePlatform,
    handleSubmit,
    handleBack,
    handleClose,
  };
}
