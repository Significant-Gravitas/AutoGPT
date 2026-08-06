"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";

import { usePostV2CreateOrganization } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { normalizeOrg } from "@/services/org-team/normalize";
import { useOrgTeamStore } from "@/services/org-team/store";

import { createOrgSchema, slugify, type CreateOrgFormValues } from "./schema";

interface Args {
  onClose: () => void;
}

export function useCreateOrgDialog({ onClose }: Args) {
  const { setOrgs, setActiveOrg } = useOrgTeamStore();

  const form = useForm<CreateOrgFormValues>({
    resolver: zodResolver(createOrgSchema),
    defaultValues: { name: "", slug: "", description: "" },
    mode: "onChange",
  });

  const { mutateAsync: createOrg, isPending } = usePostV2CreateOrganization({
    mutation: {
      onError: (error) => {
        toast({
          title: "Failed to create organization",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  function handleNameChange(name: string) {
    form.setValue("name", name, { shouldValidate: true });
    if (!form.getFieldState("slug").isDirty) {
      form.setValue("slug", slugify(name), { shouldValidate: true });
    }
  }

  async function handleSubmit(values: CreateOrgFormValues) {
    let response;
    try {
      response = await createOrg({
        data: {
          name: values.name,
          slug: values.slug,
          description: values.description || null,
        },
      });
    } catch {
      // onError already surfaced the failure toast; swallow the rejection so
      // it doesn't escape the submit handler unhandled.
      return;
    }
    const org = response.data as OrgResponse;
    setOrgs([...useOrgTeamStore.getState().orgs, normalizeOrg(org)]);
    setActiveOrg(org.id);
    getQueryClient().resetQueries();
    toast({ title: `Organization "${org.name}" created`, variant: "success" });
    handleClose();
  }

  function handleClose() {
    form.reset();
    onClose();
  }

  return {
    form,
    isPending,
    handleNameChange,
    handleSubmit,
    handleClose,
  };
}
