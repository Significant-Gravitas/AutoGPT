"use client";

import { useEffect, useMemo, useState } from "react";
import { CaretDown } from "@phosphor-icons/react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/__legacy__/ui/popover";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { groupByCreator, groupByTitle, toLlmDisplayName } from "../helpers";
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

  const creators = useMemo(() => {
    return Array.from(
      new Set(models.map((model) => model.creator)),
    ).sort((a, b) => toLlmDisplayName(a).localeCompare(toLlmDisplayName(b)));
  }, [models]);

  const modelsByCreator = useMemo(() => groupByCreator(models), [models]);

  useEffect(() => {
    if (!open) {
      return;
    }
    setView("creator");
    setActiveCreator(selectedModel?.creator ?? creators[0] ?? null);
    setActiveTitle(selectedModel?.title ?? null);
  }, [open, selectedModel?.creator, selectedModel?.title, creators]);

  const currentCreator = activeCreator ?? creators[0] ?? null;
  const currentModels = currentCreator
    ? modelsByCreator.get(currentCreator) ?? []
    : [];

  const modelsByTitle = useMemo(() => groupByTitle(currentModels), [currentModels]);

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

  const handleSelectModel = (modelName: string) => {
    onSelect(modelName);
    setOpen(false);
  };

  const triggerModel = selectedModel ?? recommendedModel ?? models[0];
  const triggerTitle = triggerModel ? triggerModel.title : "Select model";
  const triggerCreator = triggerModel?.creator ?? "";

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          disabled={disabled}
          className={cn(
            "flex w-full min-w-[15rem] items-center gap-2 rounded-lg border border-zinc-200 bg-white px-4 py-2.5 text-left",
            "hover:border-zinc-300 focus:outline-none focus:ring-2 focus:ring-zinc-200",
            disabled && "cursor-not-allowed opacity-60",
          )}
        >
          <LlmIcon value={triggerCreator} />
          <Text variant="body" className="flex-1 text-zinc-900">
            {triggerTitle}
          </Text>
          <CaretDown className="h-4 w-4 text-zinc-800" />
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={8}
        className="max-h-[60vh] w-[--radix-popover-trigger-width] overflow-y-auto rounded-md border border-zinc-200 bg-white p-0 shadow-[0px_1px_4px_rgba(12,12,13,0.12)]"
      >
        {view === "creator" && (
          <div className="flex flex-col">
            {recommendedModel && (
              <>
                <LlmMenuItem
                  title={recommendedModel.title}
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
                title={toLlmDisplayName(creator)}
                icon={<LlmIcon value={creator} />}
                showChevron={true}
                isActive={selectedModel?.creator === creator}
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
              label={toLlmDisplayName(currentCreator)}
              onBack={() => setView("creator")}
            />
            <div className="border-b border-zinc-200" />
            {modelEntries.map((entry) => (
              <LlmMenuItem
                key={entry.title}
                title={entry.title}
                icon={<LlmIcon value={currentCreator} />}
                rightSlot={<LlmPriceTier tier={entry.entries[0]?.price_tier} />}
                showChevron={entry.providerCount > 1}
                isActive={selectedModel?.title === entry.title}
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
                title={toLlmDisplayName(entry.provider)}
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
