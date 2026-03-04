"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { CaretDownIcon } from "@phosphor-icons/react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  getCreatorDisplayName,
  getModelDisplayName,
  getProviderDisplayName,
  groupByCreator,
  groupByTitle,
} from "../helpers";
import { LlmModelMetadata } from "../types";
import { LlmIcon } from "./LlmIcon";
import { LlmMenuHeader } from "./LlmMenuHeader";
import { LlmMenuItem } from "./LlmMenuItem";
import { LlmPriceTier } from "./LlmPriceTier";

type MenuView = "creator" | "model" | "provider";

type Props = {
  models: LlmModelMetadata[];
  selectedModel?: LlmModelMetadata;
  recommendedModel?: LlmModelMetadata;
  onSelect: (value: string) => void;
  disabled?: boolean;
};

export function LlmModelPicker({
  models,
  selectedModel,
  recommendedModel,
  onSelect,
  disabled,
}: Props) {
  const [open, setOpen] = useState(false);
  const [view, setView] = useState<MenuView>("creator");
  const [activeCreator, setActiveCreator] = useState<string | null>(null);
  const [activeTitle, setActiveTitle] = useState<string | null>(null);

  const modelsByCreator = useMemo(() => groupByCreator(models), [models]);

  const creators = useMemo(() => {
    return Array.from(modelsByCreator.keys()).sort((a, b) =>
      a.localeCompare(b),
    );
  }, [modelsByCreator]);

  const creatorIconValues = useMemo(() => {
    const map = new Map<string, string>();
    for (const [creator, entries] of modelsByCreator.entries()) {
      map.set(creator, entries[0]?.creator ?? creator);
    }
    return map;
  }, [modelsByCreator]);

  useEffect(() => {
    if (!open) {
      return;
    }
    setView("creator");
    setActiveCreator(
      selectedModel
        ? getCreatorDisplayName(selectedModel)
        : (creators[0] ?? null),
    );
    setActiveTitle(selectedModel ? getModelDisplayName(selectedModel) : null);
  }, [open, selectedModel, creators]);

  const currentCreator = activeCreator ?? creators[0] ?? null;

  const currentModels = useMemo(() => {
    return currentCreator ? (modelsByCreator.get(currentCreator) ?? []) : [];
  }, [currentCreator, modelsByCreator]);

  const currentCreatorIcon = useMemo(() => {
    return currentModels[0]?.creator ?? currentCreator;
  }, [currentModels, currentCreator]);

  const modelsByTitle = useMemo(
    () => groupByTitle(currentModels),
    [currentModels],
  );

  const modelEntries = useMemo(() => {
    return Array.from(modelsByTitle.entries())
      .map(([title, entries]) => {
        const providers = new Set(entries.map((entry) => entry.provider));
        return {
          title,
          entries,
          providerCount: providers.size,
        };
      })
      .sort((a, b) => a.title.localeCompare(b.title));
  }, [modelsByTitle]);

  const providerEntries = useMemo(() => {
    if (!activeTitle) {
      return [];
    }
    return modelsByTitle.get(activeTitle) ?? [];
  }, [activeTitle, modelsByTitle]);

  const handleSelectModel = useCallback(
    (modelName: string) => {
      onSelect(modelName);
      setOpen(false);
    },
    [onSelect],
  );

  const triggerModel = selectedModel ?? recommendedModel ?? models[0];
  const triggerTitle = triggerModel
    ? getModelDisplayName(triggerModel)
    : "Select model";
  const triggerCreator = triggerModel?.creator ?? "";

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "flex w-full min-w-[15rem] items-center rounded-lg border border-zinc-200 bg-white px-3 py-2 text-left",
            "hover:border-zinc-300 focus:outline-none focus:ring-2 focus:ring-zinc-200",
            disabled && "cursor-not-allowed opacity-60",
          )}
        >
          <LlmIcon value={triggerCreator} />
          <Text variant="body" className="ml-1 flex-1 text-zinc-900">
            {triggerTitle}
          </Text>
          <CaretDownIcon className="h-3 w-3 text-zinc-900" weight="bold" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={4}
        className="max-h-[45vh] w-[--radix-popover-trigger-width] min-w-[16rem] overflow-y-auto rounded-md border border-zinc-200 bg-white p-0 shadow-[0px_1px_4px_rgba(12,12,13,0.12)]"
      >
        {view === "creator" && (
          <div className="flex flex-col">
            {recommendedModel && (
              <>
                <LlmMenuItem
                  title={getModelDisplayName(recommendedModel)}
                  subtitle="Recommended"
                  icon={<LlmIcon value={recommendedModel.creator} />}
                  onClick={() => handleSelectModel(recommendedModel.name)}
                />
                <div className="border-b border-zinc-200" />
              </>
            )}
            {creators.map((creator) => (
              <LlmMenuItem
                key={creator}
                title={creator}
                icon={
                  <LlmIcon value={creatorIconValues.get(creator) ?? creator} />
                }
                showChevron={true}
                isActive={
                  selectedModel
                    ? getCreatorDisplayName(selectedModel) === creator
                    : false
                }
                onClick={() => {
                  setActiveCreator(creator);
                  setView("model");
                }}
              />
            ))}
          </div>
        )}
        {view === "model" && currentCreator && (
          <div className="flex flex-col">
            <LlmMenuHeader
              label={currentCreator}
              onBack={() => setView("creator")}
            />
            <div className="border-b border-zinc-200" />
            {modelEntries.map((entry) => (
              <LlmMenuItem
                key={entry.title}
                title={entry.title}
                icon={<LlmIcon value={currentCreatorIcon} />}
                rightSlot={<LlmPriceTier tier={entry.entries[0]?.price_tier} />}
                showChevron={entry.providerCount > 1}
                isActive={
                  selectedModel
                    ? getModelDisplayName(selectedModel) === entry.title
                    : false
                }
                onClick={() => {
                  if (entry.providerCount > 1) {
                    setActiveTitle(entry.title);
                    setView("provider");
                    return;
                  }
                  handleSelectModel(entry.entries[0].name);
                }}
              />
            ))}
          </div>
        )}
        {view === "provider" && activeTitle && (
          <div className="flex flex-col">
            <LlmMenuHeader
              label={activeTitle}
              onBack={() => setView("model")}
            />
            <div className="border-b border-zinc-200" />
            {providerEntries.map((entry) => (
              <LlmMenuItem
                key={`${entry.title}-${entry.provider}`}
                title={getProviderDisplayName(entry)}
                icon={<LlmIcon value={entry.provider} />}
                isActive={selectedModel?.provider === entry.provider}
                onClick={() => handleSelectModel(entry.name)}
              />
            ))}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}
