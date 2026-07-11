"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";

import {
  useGetV2ListOrganizationAliases,
  usePostV2CreateOrganizationAlias,
} from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgAliasResponse } from "@/app/api/__generated__/models/orgAliasResponse";
import { toast } from "@/components/molecules/Toast/use-toast";

import { createAliasSchema, type CreateAliasFormValues } from "./schema";

interface Args {
  orgId: string;
  isAdmin: boolean;
}

export function useAliasesSection({ orgId, isAdmin }: Args) {
  const aliasesQuery = useGetV2ListOrganizationAliases(orgId, {
    query: {
      enabled: Boolean(orgId),
      select: (res) => res.data as OrgAliasResponse[],
    },
  });

  const form = useForm<CreateAliasFormValues>({
    resolver: zodResolver(createAliasSchema),
    defaultValues: { alias_slug: "" },
    mode: "onChange",
  });

  const { mutateAsync: createAlias, isPending: isCreating } =
    usePostV2CreateOrganizationAlias({
      mutation: {
        onError: (error) => {
          toast({
            title: "Failed to add alias",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  async function handleCreate(values: CreateAliasFormValues) {
    await createAlias({ orgId, data: { alias_slug: values.alias_slug } });
    toast({ title: `Alias "${values.alias_slug}" added`, variant: "success" });
    form.reset();
    aliasesQuery.refetch();
  }

  return {
    form,
    aliases: aliasesQuery.data ?? [],
    isLoading: aliasesQuery.isLoading,
    canManage: isAdmin,
    isCreating,
    handleCreate,
  };
}
