"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowUpIcon } from "@phosphor-icons/react";
import { useCopilotHome } from "./useCopilotHome";

export default function CopilotPage() {
  const {
    greetingName,
    value,
    quickActions,
    isFlagReady,
    isChatEnabled,
    handleChange,
    handleSubmit,
    handleKeyDown,
    handleQuickAction,
  } = useCopilotHome();

  if (!isFlagReady || isChatEnabled === false) {
    return null;
  }

  return (
    <div className="flex min-h-full flex-1 items-center justify-center px-6 py-10">
      <div className="w-full max-w-2xl text-center">
        <Text variant="h2" className="mb-3 text-zinc-700">
          Hey, <span className="text-violet-600">{greetingName}</span>
        </Text>
        <Text variant="h3" className="mb-8 text-zinc-900">
          What do you want to automate?
        </Text>

        <form onSubmit={handleSubmit} className="mb-8">
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
              className="min-h-[3.5rem] pr-12 text-base"
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

        <div className="flex flex-wrap items-center justify-center gap-3">
          {quickActions.map((action) => (
            <Button
              key={action}
              variant="outline"
              size="small"
              onClick={() => handleQuickAction(action)}
              className="border-zinc-300 text-zinc-700 hover:border-zinc-400 hover:bg-zinc-50"
            >
              {action}
            </Button>
          ))}
        </div>
      </div>
    </div>
  );
}
