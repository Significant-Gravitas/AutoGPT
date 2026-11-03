"use client";

import {
  getGetAdminGetOrganizationAccessQueryKey,
  getGetAdminListOrganizationAccessQueryKey,
  useDeleteAdminDisableOrganizationAccess,
  useGetAdminGetOrganizationAccess,
  usePutAdminEnableOrganizationAccess,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { getErrorMessage } from "../../helpers";

export function useOrgAccessControl(userId: string) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const statusQuery = useGetAdminGetOrganizationAccess(userId, {
    query: { select: okData },
  });
  const enableMutation = usePutAdminEnableOrganizationAccess();
  const disableMutation = useDeleteAdminDisableOrganizationAccess();

  function refresh() {
    void queryClient.invalidateQueries({
      queryKey: getGetAdminGetOrganizationAccessQueryKey(userId),
    });
    void queryClient.invalidateQueries({
      queryKey: getGetAdminListOrganizationAccessQueryKey().slice(0, 1),
    });
  }

  async function run(action: () => Promise<unknown>, failure: string) {
    try {
      await action();
    } catch (error) {
      toast({
        title: failure,
        description: getErrorMessage(error),
        variant: "destructive",
      });
      return false;
    } finally {
      refresh();
    }
    return true;
  }

  return {
    status: statusQuery.data,
    isLoading: statusQuery.isLoading,
    retry: () => void statusQuery.refetch(),
    enable: () =>
      run(
        () => enableMutation.mutateAsync({ userId, data: { note: null } }),
        "Couldn't turn organizations on",
      ),
    disable: () =>
      run(
        () => disableMutation.mutateAsync({ userId }),
        "Couldn't turn organizations off",
      ),
    isSaving: enableMutation.isPending || disableMutation.isPending,
  };
}
