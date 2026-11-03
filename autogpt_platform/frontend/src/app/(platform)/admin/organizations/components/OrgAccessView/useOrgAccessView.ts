"use client";

import {
  getGetAdminListOrganizationAccessQueryKey,
  useDeleteAdminDisableOrganizationAccess,
  useGetAdminListOrganizationAccess,
  usePostAdminEnableOrganizationAccessByEmail,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { getErrorMessage, PAGE_SIZE } from "../../helpers";

interface EnableArgs {
  email: string;
  note: string;
}

export function useOrgAccessView() {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const params = {
    search: search.trim() || undefined,
    page,
    page_size: PAGE_SIZE,
  };
  const grantsQuery = useGetAdminListOrganizationAccess(params, {
    query: { select: okData },
  });
  const enableMutation = usePostAdminEnableOrganizationAccessByEmail();
  const disableMutation = useDeleteAdminDisableOrganizationAccess();

  function refresh() {
    void queryClient.invalidateQueries({
      queryKey: getGetAdminListOrganizationAccessQueryKey().slice(0, 1),
    });
  }

  function changeSearch(value: string) {
    setSearch(value);
    setPage(1);
  }

  async function enable({ email, note }: EnableArgs) {
    try {
      await enableMutation.mutateAsync({
        data: { email: email.trim(), note: note.trim() || null },
      });
    } catch (error) {
      toast({
        title: "Couldn't turn organizations on",
        description: getErrorMessage(error),
        variant: "destructive",
      });
      return false;
    }
    refresh();
    toast({
      title: `Organizations are on for ${email.trim()}`,
      variant: "success",
    });
    return true;
  }

  async function disable(userId: string, email: string) {
    try {
      await disableMutation.mutateAsync({ userId });
    } catch (error) {
      toast({
        title: "Couldn't turn organizations off",
        description: getErrorMessage(error),
        variant: "destructive",
      });
      return false;
    }
    refresh();
    toast({ title: `Organizations are off for ${email}`, variant: "success" });
    return true;
  }

  return {
    grants: grantsQuery.data?.grants ?? [],
    total: grantsQuery.data?.total ?? 0,
    isLoading: grantsQuery.isLoading,
    isError: grantsQuery.isError,
    error: grantsQuery.error,
    retry: () => void grantsQuery.refetch(),
    search,
    changeSearch,
    page,
    setPage,
    enable,
    isEnabling: enableMutation.isPending,
    disable,
    isDisabling: disableMutation.isPending,
  };
}
