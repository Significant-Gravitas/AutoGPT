"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { usePostV2ConvertPersonalOrgToTeamOrg } from "@/app/api/__generated__/endpoints/orgs/orgs";
import type { OrgResponse } from "@/app/api/__generated__/models/orgResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useOrgTeamStore } from "@/services/org-team/store";

import { getOrgsAfterConvert } from "./helpers";

interface Args {
  org: OrgResponse;
}

export function useConvertOrgSection({ org }: Args) {
  const queryClient = useQueryClient();
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);

  const { mutateAsync: convertOrg, isPending: isConverting } =
    usePostV2ConvertPersonalOrgToTeamOrg({
      mutation: {
        onError: (error) => {
          toast({
            title: "Couldn't convert this organization",
            description:
              error instanceof Error ? error.message : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  async function handleConvertConfirmed() {
    let converted: OrgResponse;
    try {
      const response = await convertOrg({ orgId: org.id });
      if (response.status !== 200) return;
      converted = response.data;
    } catch {
      // onError already surfaced the failure toast.
      return;
    }

    const store = useOrgTeamStore.getState();
    store.setOrgs(await getOrgsAfterConvert(converted, store.orgs));
    // The converted organization keeps its id, so the viewer is usually in it
    // already. Re-selecting the same id would mark the context unloaded with
    // nothing left to reload it, so only switch when the id differs.
    if (useOrgTeamStore.getState().activeOrgID !== converted.id) {
      useOrgTeamStore.getState().setActiveOrg(converted.id);
    }
    void queryClient.resetQueries();
    toast({
      title: `${converted.name} is now a team organization`,
      variant: "success",
    });
    setIsConfirmOpen(false);
  }

  return {
    isConfirmOpen,
    setIsConfirmOpen,
    isConverting,
    handleConvertConfirmed,
  };
}
