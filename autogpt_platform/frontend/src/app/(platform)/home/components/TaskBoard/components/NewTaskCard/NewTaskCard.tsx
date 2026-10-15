import { HOME_COMPOSER_ID } from "@/app/(platform)/copilot/components/EmptySession/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Add01Icon } from "@hugeicons/core-free-icons";

export function NewTaskCard() {
  function handleClick() {
    const composer = document.getElementById(HOME_COMPOSER_ID);
    composer?.scrollIntoView({ behavior: "smooth", block: "center" });
    composer?.focus({ preventScroll: true });
  }

  return (
    <button
      type="button"
      onClick={handleClick}
      className="group flex w-full items-center gap-3 rounded-lg border border-dashed border-zinc-300 px-3.5 py-3 text-left outline-none transition-colors hover:border-zinc-400 hover:bg-white focus-visible:ring-2 focus-visible:ring-violet-400"
    >
      <span className="flex size-8 shrink-0 items-center justify-center rounded-full border border-zinc-200 bg-white text-zinc-500 transition-colors group-hover:text-zinc-900">
        <Icon icon={Add01Icon} size={16} />
      </span>
      <span className="min-w-0">
        <Text variant="body-medium" tone="primary" as="span" className="block">
          New task
        </Text>
        <Text variant="small" tone="muted" as="span" className="block">
          Tell Otto or an expert what to work on
        </Text>
      </span>
    </button>
  );
}
