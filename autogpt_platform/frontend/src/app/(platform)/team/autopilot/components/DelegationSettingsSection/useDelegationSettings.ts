import { useQueryClient } from "@tanstack/react-query";
import {
  getGetDelegationSettingsQueryKey,
  useGetDelegationSettings,
  useUpdateDelegationSettings,
} from "@/app/api/__generated__/endpoints/experts/experts";
import type { DelegationSettingsUpdate } from "@/app/api/__generated__/models/delegationSettingsUpdate";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { withDefaults } from "./helpers";

interface Args {
  enabled: boolean;
}

type CachedSettings = { status: number; data: unknown } | undefined;

export function useDelegationSettings({ enabled }: Args) {
  const queryClient = useQueryClient();
  const queryKey = getGetDelegationSettingsQueryKey();
  const query = useGetDelegationSettings({
    query: { select: (res) => okData(res) ?? null, enabled },
  });
  const { mutateAsync, isPending } = useUpdateDelegationSettings();
  const settings = withDefaults(query.data);

  async function update(patch: Partial<DelegationSettingsUpdate>) {
    const next = { ...settings, ...patch };
    await queryClient.cancelQueries({ queryKey });
    const previous = queryClient.getQueryData<CachedSettings>(queryKey);
    if (previous)
      queryClient.setQueryData(queryKey, { ...previous, data: next });
    try {
      const response = await mutateAsync({ data: next });
      if (response.status !== 200) throw new Error("Not saved");
      queryClient.setQueryData(queryKey, response);
    } catch {
      queryClient.setQueryData(queryKey, previous);
      toast({
        title: "Couldn't save the delegation settings",
        description: "Your last change was undone. Try again.",
        variant: "destructive",
      });
    } finally {
      await queryClient.invalidateQueries({ queryKey });
    }
  }

  return {
    settings,
    isLoading: query.isLoading,
    isError: query.isError && !query.data,
    isSaving: isPending,
    refetch: query.refetch,
    update,
  };
}
