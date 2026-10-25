import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import {
  getGetDelegationSettingsQueryKey,
  type getDelegationSettingsResponse,
  useGetDelegationSettings,
  useUpdateDelegationSettings,
} from "@/app/api/__generated__/endpoints/experts/experts";
import type { DelegationSettingsUpdate } from "@/app/api/__generated__/models/delegationSettingsUpdate";
import { okData } from "@/app/api/helpers";
import { toast } from "@/components/molecules/Toast/use-toast";
import { withDefaults } from "./helpers";

type Patch = Partial<DelegationSettingsUpdate>;

interface PendingChange {
  id: number;
  patch: Patch;
}

/** Otto's delegation settings, saved one change at a time.
 *
 *  The server takes the whole settings object, so two changes saved side by
 *  side would each overwrite the other. The query cache holds only what the
 *  server confirmed; changes not yet saved are laid over it for display, and
 *  saved in order, each on top of the settings the save before confirmed. A
 *  change the server refuses is dropped on its own: the others stay. */
export function useDelegationSettings() {
  const queryClient = useQueryClient();
  const queryKey = getGetDelegationSettingsQueryKey();
  const query = useGetDelegationSettings({
    query: { select: (res) => okData(res) ?? null },
  });
  const { mutateAsync } = useUpdateDelegationSettings();
  const [pending, setPending] = useState<PendingChange[]>([]);
  const queue = useRef<Promise<void>>(Promise.resolve());
  const nextId = useRef(0);

  const settings = pending.reduce(
    (shown, change) => ({ ...shown, ...change.patch }),
    withDefaults(query.data),
  );

  function confirmed() {
    const cached =
      queryClient.getQueryData<getDelegationSettingsResponse>(queryKey);
    return withDefaults(okData(cached) ?? null);
  }

  async function save(change: PendingChange) {
    // A read still on its way could land after this save and put the older
    // settings back in the cache the next save builds on.
    await queryClient.cancelQueries({ queryKey });
    try {
      const response = await mutateAsync({
        data: { ...confirmed(), ...change.patch },
      });
      if (response.status !== 200) throw new Error("Not saved");
      queryClient.setQueryData(queryKey, response);
    } catch {
      toast({
        title: "Couldn't save the delegation settings",
        description: "Your last change was undone. Try again.",
        variant: "destructive",
      });
    } finally {
      setPending((all) => all.filter((other) => other.id !== change.id));
    }
  }

  function update(patch: Patch) {
    const change = { id: nextId.current++, patch };
    setPending((all) => [...all, change]);
    const saved = queue.current.then(() => save(change));
    queue.current = saved;
    return saved;
  }

  return {
    settings,
    isLoading: query.isLoading,
    isError: query.isError && !query.data,
    isSaving: pending.length > 0,
    refetch: query.refetch,
    update,
  };
}
