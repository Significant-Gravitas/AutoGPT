"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";

import { usePostV2CreateOrganization } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";
import { useOrgTeamStore } from "@/services/org-team/store";

import { createOrgSchema, slugify, type CreateOrgFormValues } from "./schema";

interface Args {
  onClose: () => void;
}

export function useCreateOrgDialog({ onClose }: Args) {
  const { orgs, setOrgs, setActiveOrg } = useOrgTeamStore();

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
    const response = await createOrg({
      data: {
        name: values.name,
        slug: values.slug,
        description: values.description || null,
      },
    });
    const org = response.data as OrgResponse;
    setOrgs([
      ...orgs,
      {
        id: org.id,
        name: org.name,
        slug: org.slug,
        avatarUrl: org.avatar_url ?? null,
        isPersonal: org.is_personal,
        memberCount: org.member_count,
      },
    ]);
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
