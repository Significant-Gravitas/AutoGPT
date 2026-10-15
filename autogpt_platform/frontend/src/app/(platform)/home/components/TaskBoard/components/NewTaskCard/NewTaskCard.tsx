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
      <span className="flex size-9 shrink-0 items-center justify-center rounded-full border border-white bg-gradient-to-b from-white/90 to-zinc-100/70 text-zinc-700 shadow-[inset_0_1px_1px_rgba(255,255,255,0.9),0_1px_3px_rgba(0,0,0,0.08),0_0_0_1px_rgba(0,0,0,0.04)] backdrop-blur-md transition-[transform,color] duration-200 ease-out group-hover:scale-105 group-hover:text-zinc-900 motion-reduce:transition-none">
        <Icon icon={Add01Icon} size={16} strokeWidth={2.25} />
      </span>
      <span className="min-w-0">
        <Text variant="body-medium" tone="primary" as="span" className="block">
          New task
        </Text>
        <Text variant="small" tone="muted" as="span" className="block truncate">
          Tell Otto or an expert what to do
        </Text>
      </span>
    </button>
  );
}
