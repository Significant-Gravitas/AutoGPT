"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { usePatchV2UpdateOrganization } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useOrgTeamStore } from "@/services/org-team/store";

const orgProfileSchema = z.object({
  name: z
    .string()
    .trim()
    .min(1, "Name is required")
    .max(100, "Name must be 100 characters or less"),
  slug: z
    .string()
    .trim()
    .min(3, "Slug must be at least 3 characters")
    .max(50, "Slug must be 50 characters or less")
    .regex(
      /^[a-z0-9]+(?:-[a-z0-9]+)*$/,
      "Lowercase letters, numbers and dashes only",
    ),
  description: z
    .string()
    .trim()
    .max(500, "Description must be 500 characters or less")
    .optional(),
});

export type OrgProfileFormValues = z.infer<typeof orgProfileSchema>;

interface Args {
  org: OrgResponse;
  onSaved: () => void;
}

export function useOrgProfileSection({ org, onSaved }: Args) {
  const { orgs, setOrgs } = useOrgTeamStore();

  const form = useForm<OrgProfileFormValues>({
    resolver: zodResolver(orgProfileSchema),
    defaultValues: {
      name: org.name,
      slug: org.slug,
      description: org.description ?? "",
    },
    mode: "onChange",
  });

  const { mutateAsync: updateOrg, isPending } = usePatchV2UpdateOrganization({
    mutation: {
      onError: (error) => {
        toast({
          title: "Failed to update organization",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  async function handleSubmit(values: OrgProfileFormValues) {
    const response = await updateOrg({
      orgId: org.id,
      data: {
        name: values.name !== org.name ? values.name : undefined,
        slug: values.slug !== org.slug ? values.slug : undefined,
        description:
          (values.description || "") !== (org.description ?? "")
            ? values.description || null
            : undefined,
      },
    });
    const updated = response.data as OrgResponse;
    // Keep the switcher's org list in step with the rename.
    setOrgs(
      orgs.map((o) =>
        o.id === updated.id
          ? { ...o, name: updated.name, slug: updated.slug }
          : o,
      ),
    );
    form.reset({
      name: updated.name,
      slug: updated.slug,
      description: updated.description ?? "",
    });
    toast({ title: "Organization updated", variant: "success" });
    onSaved();
  }

  return {
    form,
    isPending,
    isDirty: form.formState.isDirty,
    handleSubmit,
  };
}
