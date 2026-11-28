import { useQuery, useQueryClient } from "@tanstack/react-query";
import { router } from "expo-router";
import { useRef, useState } from "react";
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
import { sessionGeneration } from "@/lib/session";

import { eraseWarning, factText, understandingRows, withFactBack, withoutFact } from "./helpers";

const EMPTY_UNDERSTANDING = { updated_at: null } satisfies Partial<MemoryUnderstanding>;
let eraseVersion = 0;

export function useMemoryScreen() {
  const queryClient = useQueryClient();
  const { toast, show } = useToast();
  const [isErasing, setIsErasing] = useState(false);
  const erasing = useRef(false);
  const pendingForgets = useRef(0);
  const understanding = useQuery({
    queryKey: memoryKeys.understanding,
    queryFn: getUnderstanding,
  });
  const facts = useQuery({ queryKey: memoryKeys.facts, queryFn: () => listMemoryFacts() });
  const overview = useQuery({ queryKey: memoryKeys.overview, queryFn: getMemoryOverview });
  const queries = [understanding, facts, overview];

  async function openFactMenu(fact: MemoryFact) {
    const generation = sessionGeneration();
    tapMenu();
    const action = await chooseAction(factText(fact), [
      { id: "forget", label: "Forget", isDestructive: true },
    ]);
    if (action === "forget" && generation === sessionGeneration()) void forgetFact(fact.uuid);
  }

  async function forgetFact(uuid: string) {
    if (erasing.current) return;
    const generation = sessionGeneration();
    const version = eraseVersion;
    pendingForgets.current += 1;
    try {
      await queryClient.cancelQueries({ queryKey: memoryKeys.facts });
      if (generation !== sessionGeneration() || version !== eraseVersion || erasing.current) return;
      const before = queryClient.getQueryData<MemoryFactList>(memoryKeys.facts);
      if (!before?.items.some((item) => item.uuid === uuid)) return;
      queryClient.setQueryData<MemoryFactList>(memoryKeys.facts, withoutFact(before, uuid));
      try {
        await forgetMemoryFact(uuid);
      } catch {
        if (generation !== sessionGeneration() || version !== eraseVersion) return;
        queryClient.setQueryData<MemoryFactList>(memoryKeys.facts, (now) =>
          withFactBack(before, now, uuid),
        );
        show({ message: "Couldn't forget that memory. Try again.", tone: "error" });
      }
    } finally {
      pendingForgets.current -= 1;
      if (
        generation === sessionGeneration() &&
        version === eraseVersion &&
        !erasing.current &&
        pendingForgets.current === 0
      ) {
        void queryClient.invalidateQueries({ queryKey: memoryKeys.all });
      }
    }
  }

  function confirmForgetUnderstanding() {
    const generation = sessionGeneration();
    Alert.alert(
      "Forget what Otto knows about your work?",
      "Otto stops starting new chats with this. Your recent memories stay.",
      [
        { text: "Cancel", style: "cancel" },
        {
          text: "Forget",
          style: "destructive",
          onPress: () => {
            if (generation === sessionGeneration()) void removeUnderstanding();
          },
        },
      ],
    );
  }

  async function removeUnderstanding() {
    if (erasing.current) return;
    const generation = sessionGeneration();
    const version = eraseVersion;
    await queryClient.cancelQueries({ queryKey: memoryKeys.understanding });
    if (generation !== sessionGeneration() || version !== eraseVersion || erasing.current) return;
    const before = queryClient.getQueryData<MemoryUnderstanding>(memoryKeys.understanding);
    queryClient.setQueryData(memoryKeys.understanding, EMPTY_UNDERSTANDING);
    try {
      await forgetUnderstanding();
    } catch {
      if (generation !== sessionGeneration() || version !== eraseVersion) return;
      queryClient.setQueryData(memoryKeys.understanding, before);
      show({ message: "Couldn't forget this. Try again.", tone: "error" });
    } finally {
      if (generation === sessionGeneration() && version === eraseVersion && !erasing.current) {
        void queryClient.invalidateQueries({ queryKey: memoryKeys.understanding });
      }
    }
  }

  function confirmErase() {
    if (!overview.data || overview.isError || erasing.current) return;
    const generation = sessionGeneration();
    Alert.alert("Erase all memory?", eraseWarning(overview.data), [
      { text: "Cancel", style: "cancel" },
      {
        text: "Erase",
        style: "destructive",
        onPress: () => {
          if (generation === sessionGeneration()) void erase();
        },
      },
    ]);
  }

  async function erase() {
    if (erasing.current) return;
    const generation = sessionGeneration();
    erasing.current = true;
    setIsErasing(true);
    try {
      const result = await eraseMemory();
      if (generation !== sessionGeneration()) return;
      if (!result.erased) throw new Error("Memory erase did not complete");
      eraseVersion += 1;
      await queryClient.cancelQueries({ queryKey: memoryKeys.all });
      if (generation !== sessionGeneration()) return;
      queryClient.setQueryData(memoryKeys.facts, { items: [] } satisfies MemoryFactList);
      queryClient.setQueryData(memoryKeys.understanding, EMPTY_UNDERSTANDING);
      queryClient.setQueryData(memoryKeys.overview, { facts: 0, entities: 0, episodes: 0 });
      show({ message: "Memory erased.", tone: "success" });
    } catch {
      if (generation === sessionGeneration()) {
        show({ message: "Couldn't erase memory. Try again.", tone: "error" });
      }
    } finally {
      erasing.current = false;
      setIsErasing(false);
      if (generation === sessionGeneration()) {
        void queryClient.invalidateQueries({ queryKey: memoryKeys.all });
      }
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
    isOverviewError: overview.isError,
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
