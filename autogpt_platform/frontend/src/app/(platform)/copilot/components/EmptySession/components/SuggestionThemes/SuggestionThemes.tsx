"use client";

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Button } from "@/components/atoms/Button/Button";
import {
  BookOpenIcon,
  PaintBrushIcon,
  LightningIcon,
  ListChecksIcon,
  SpinnerGapIcon,
} from "@phosphor-icons/react";
import { useState } from "react";
import type { SuggestionTheme } from "../../helpers";

const THEME_ICONS: Record<string, typeof BookOpenIcon> = {
  Learn: BookOpenIcon,
  Create: PaintBrushIcon,
  Automate: LightningIcon,
  Organize: ListChecksIcon,
};

interface Props {
  themes: SuggestionTheme[];
  onSend: (prompt: string) => void | Promise<void>;
  disabled?: boolean;
}

export function SuggestionThemes({ themes, onSend, disabled }: Props) {
  const [openTheme, setOpenTheme] = useState<string | null>(null);
  const [loadingPrompt, setLoadingPrompt] = useState<string | null>(null);

  async function handlePromptClick(theme: string, prompt: string) {
    if (disabled || loadingPrompt) return;
    setLoadingPrompt(`${theme}:${prompt}`);
    try {
      await onSend(prompt);
    } finally {
      setLoadingPrompt(null);
      setOpenTheme(null);
    }
  }

  return (
    <div className="flex flex-wrap items-center justify-center gap-3">
      {themes.map((theme) => {
        const Icon = THEME_ICONS[theme.name];
        return (
          <Popover
            key={theme.name}
            open={openTheme === theme.name}
            onOpenChange={(open) => setOpenTheme(open ? theme.name : null)}
          >
            <PopoverTrigger asChild>
              <Button
                type="button"
                variant="outline"
                size="small"
                disabled={disabled || loadingPrompt !== null}
                className="shrink-0 gap-2 border-zinc-300 px-3 py-2 text-[.9rem] text-zinc-600"
              >
                {Icon && <Icon size={16} weight="regular" />}
                {theme.name}
              </Button>
            </PopoverTrigger>
            <PopoverContent align="center" className="w-80 p-2">
              <ul className="grid gap-0.5">
                {theme.prompts.map((prompt) => (
                  <li key={prompt}>
                    <button
                      type="button"
                      disabled={disabled || loadingPrompt !== null}
                      onClick={() => void handlePromptClick(theme.name, prompt)}
                      className="w-full rounded-md px-3 py-2 text-left text-sm text-zinc-700 transition-colors hover:bg-zinc-100 disabled:opacity-50"
                    >
                      {loadingPrompt === `${theme.name}:${prompt}` ? (
                        <span className="flex items-center gap-2">
                          <SpinnerGapIcon
                            className="h-4 w-4 animate-spin"
                            weight="bold"
                          />
                          {prompt}
                        </span>
                      ) : (
                        prompt
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            </PopoverContent>
          </Popover>
        );
      })}
    </div>
  );
}
