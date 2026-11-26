import { useQuery, useQueryClient } from "@tanstack/react-query";
import { router } from "expo-router";
import { useState } from "react";
import { Alert } from "react-native";

import { useToast } from "@/components/useToast";
import { chooseAction } from "@/lib/actionMenu";
import { tapMenu } from "@/lib/haptics";
import {
  eraseMemory,
  forgetMemoryFact,
  forgetUnderstanding,
  getMemoryOverview,
  getUnderstanding,
  listMemoryFacts,
  type MemoryFact,
  type MemoryFactList,
  type MemoryUnderstanding,
  memoryKeys,
} from "@/lib/api/memory";

import { eraseWarning, factText, understandingRows, withFactBack, withoutFact } from "./helpers";

export function useMemoryScreen() {
  const queryClient = useQueryClient();
  const { toast, show } = useToast();
  const [isErasing, setIsErasing] = useState(false);
  const understanding = useQuery({
    queryKey: memoryKeys.understanding,
    queryFn: getUnderstanding,
  });
  const facts = useQuery({ queryKey: memoryKeys.facts, queryFn: () => listMemoryFacts() });
  const overview = useQuery({ queryKey: memoryKeys.overview, queryFn: getMemoryOverview });
  const queries = [understanding, facts, overview];

  async function openFactMenu(fact: MemoryFact) {
    tapMenu();
    const action = await chooseAction(factText(fact), [
      { id: "forget", label: "Forget", isDestructive: true },
    ]);
    if (action === "forget") void forgetFact(fact.uuid);
  }

  async function forgetFact(uuid: string) {
    await queryClient.cancelQueries({ queryKey: memoryKeys.facts });
    const before = queryClient.getQueryData<MemoryFactList>(memoryKeys.facts);
    if (!before) return;
    queryClient.setQueryData<MemoryFactList>(memoryKeys.facts, withoutFact(before, uuid));
    try {
      await forgetMemoryFact(uuid);
    } catch {
      queryClient.setQueryData<MemoryFactList>(memoryKeys.facts, (now) =>
        withFactBack(before, now, uuid),
      );
      show({ message: "Couldn't forget that memory. Try again.", tone: "error" });
    } finally {
      void queryClient.invalidateQueries({ queryKey: memoryKeys.all });
    }
  }

  function confirmForgetUnderstanding() {
    Alert.alert(
      "Forget what Otto knows about your work?",
      "Otto stops starting new chats with this. Your recent memories stay.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Forget",
          style: "destructive",
          onPress: () => void removeUnderstanding(),
        },
      ],
    );
  }

  async function removeUnderstanding() {
    await queryClient.cancelQueries({ queryKey: memoryKeys.understanding });
    const before = queryClient.getQueryData<MemoryUnderstanding>(memoryKeys.understanding);
    queryClient.setQueryData<Partial<MemoryUnderstanding>>(memoryKeys.understanding, {
      updated_at: null,
    });
    try {
      await forgetUnderstanding();
    } catch {
      queryClient.setQueryData(memoryKeys.understanding, before);
      show({ message: "Couldn't forget this. Try again.", tone: "error" });
    } finally {
      void queryClient.invalidateQueries({ queryKey: memoryKeys.understanding });
    }
  }

  function confirmErase() {
    if (!overview.data) return;
    Alert.alert("Erase all memory?", eraseWarning(overview.data), [
      { text: "Cancel", style: "cancel" },
      { text: "Erase", style: "destructive", onPress: () => void erase() },
    ]);
  }

  async function erase() {
    setIsErasing(true);
    try {
      await eraseMemory();
      show({ message: "Memory erased.", tone: "success" });
    } catch {
      show({ message: "Couldn't erase memory. Try again.", tone: "error" });
    } finally {
      setIsErasing(false);
      void queryClient.invalidateQueries({ queryKey: memoryKeys.all });
    }
  }

  function retry() {
    queries.filter((query) => query.isError).forEach((query) => void query.refetch());
  }

  function goBack() {
    router.navigate("/settings");
  }

  return {
    understandingRows: understanding.data ? understandingRows(understanding.data) : null,
    facts: facts.data?.items ?? null,
    overview: overview.data ?? null,
    isLoading: queries.every((query) => query.isPending),
    isUnderstandingPending: understanding.isPending,
    isFactsPending: facts.isPending,
    isLoadError: queries.every((query) => query.isError && !query.data),
    isRetrying: queries.some((query) => query.isFetching),
    isErasing,
    retry,
    goBack,
    openFactMenu,
    forgetFact,
    confirmForgetUnderstanding,
    confirmErase,
    toast,
  };
}
