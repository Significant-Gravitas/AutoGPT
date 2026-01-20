"use client";

import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowUpIcon } from "@phosphor-icons/react";
import { useEffect } from "react";
import { useCopilotHome } from "./useCopilotHome";

export default function CopilotPage() {
  const {
    greetingName,
    value,
    quickActions,
    isFlagReady,
    isChatEnabled,
    isUserLoading,
    isLoggedIn,
    handleChange,
    handleSubmit,
    handleKeyDown,
    handleQuickAction,
  } = useCopilotHome();

  useEffect(() => {
    const textarea = document.getElementById(
      "copilot-prompt",
    ) as HTMLTextAreaElement;
    if (!textarea) return;
    textarea.style.height = "auto";
    const lineHeight = parseInt(
      window.getComputedStyle(textarea).lineHeight,
      10,
    );
    const maxRows = 5;
    const maxHeight = lineHeight * maxRows;
    const newHeight = Math.min(textarea.scrollHeight, maxHeight);
    textarea.style.height = `${newHeight}px`;
    textarea.style.overflowY =
      textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [value]);

  if (!isFlagReady || isChatEnabled === false || !isLoggedIn) {
    return null;
  }

  const isLoading = isUserLoading;

  return (
    <div className="flex h-full flex-1 items-center justify-center overflow-y-auto bg-[#f8f8f9] px-6 py-10">
      <div className="w-full text-center">
        {isLoading ? (
          <div className="mx-auto max-w-2xl">
            <Skeleton className="mx-auto mb-3 h-8 w-64" />
            <Skeleton className="mx-auto mb-8 h-6 w-80" />
            <div className="mb-8">
              <Skeleton className="mx-auto h-14 w-full rounded-lg" />
            </div>
            <div className="flex flex-wrap items-center justify-center gap-3">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-9 w-48 rounded-md" />
              ))}
            </div>
          </div>
        ) : (
          <>
            <div className="mx-auto max-w-2xl">
              <Text
                variant="h3"
                className="mb-3 !text-[1.375rem] text-zinc-700"
              >
                Hey, <span className="text-violet-600">{greetingName}</span>
              </Text>
              <Text variant="h3" className="mb-8 !font-normal">
                What do you want to automate?
              </Text>

              <form onSubmit={handleSubmit} className="mb-6">
                <div className="relative">
                  <Input
                    id="copilot-prompt"
                    label="Copilot prompt"
                    hideLabel
                    type="textarea"
                    value={value}
                    onChange={handleChange}
                    onKeyDown={handleKeyDown}
                    rows={1}
                    placeholder='You can search or just ask - e.g. "create a blog post outline"'
                    wrapperClassName="mb-0"
                    className="!rounded-full border-transparent !py-5 pr-12 !text-[1rem] [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
                  />
                  <Button
                    type="submit"
                    variant="icon"
                    size="icon"
                    aria-label="Submit prompt"
                    className="absolute right-2 top-1/2 -translate-y-1/2 border-zinc-800 bg-zinc-800 text-white hover:border-zinc-900 hover:bg-zinc-900"
                    disabled={!value.trim()}
                  >
                    <ArrowUpIcon className="h-4 w-4" weight="bold" />
                  </Button>
                </div>
              </form>
            </div>
            <div className="flex flex-nowrap items-center justify-center gap-3 overflow-x-auto [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
              {quickActions.map((action) => (
                <Button
                  key={action}
                  variant="outline"
                  size="small"
                  onClick={() => handleQuickAction(action)}
                  className="h-auto shrink-0 border-zinc-600 !px-4 !py-2 text-[1rem] text-zinc-600"
                >
                  {action}
                </Button>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
