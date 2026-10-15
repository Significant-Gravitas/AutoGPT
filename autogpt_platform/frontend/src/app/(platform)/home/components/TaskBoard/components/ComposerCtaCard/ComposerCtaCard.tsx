import { GLASS_CIRCLE_CLASS } from "@/app/(platform)/copilot/components/ChatInput/helpers";
import { HOME_COMPOSER_ID } from "@/app/(platform)/copilot/components/EmptySession/helpers";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Add01Icon } from "@hugeicons/core-free-icons";

interface Props {
  title: string;
  description: string;
  /** Drafted into the Home composer; without it the composer just gets focus. */
  prompt?: string;
}

export function ComposerCtaCard({ title, description, prompt }: Props) {
  const setInitialPrompt = useCopilotUIStore((s) => s.setInitialPrompt);

  function handleClick() {
    const composer = document.getElementById(HOME_COMPOSER_ID);
    composer?.scrollIntoView({ behavior: "smooth", block: "center" });
    if (prompt) setInitialPrompt(prompt);
    else composer?.focus({ preventScroll: true });
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      className="group flex w-full items-center gap-3 rounded-lg border border-dashed border-zinc-300 px-3.5 py-3 text-left outline-none transition-colors hover:border-zinc-400 hover:bg-white focus-visible:ring-2 focus-visible:ring-violet-400"
    >
      <span
        className={cn(
          "flex size-9 shrink-0 items-center justify-center transition-[transform,color] duration-200 ease-out group-hover:scale-105 group-hover:text-zinc-900 motion-reduce:transition-none",
          GLASS_CIRCLE_CLASS,
        )}
      >
        <Icon icon={Add01Icon} size={16} strokeWidth={2.25} />
      </span>
      <span className="min-w-0">
        <Text variant="body-medium" tone="primary" as="span" className="block">
          {title}
        </Text>
        <Text variant="small" tone="muted" as="span" className="block truncate">
          {description}
        </Text>
      </span>
    </button>
  );
}
