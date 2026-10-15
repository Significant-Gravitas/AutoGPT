import { HOME_COMPOSER_ID } from "@/app/(platform)/copilot/components/EmptySession/helpers";
import { SMALL_ORB_PARAMS } from "@/app/(platform)/copilot/helpers/greetingOrb";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { GlassOrb } from "@/components/molecules/GlassOrb/GlassOrb";
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
      <span className="relative block size-9 shrink-0 transition-transform duration-200 ease-out group-hover:scale-110 motion-reduce:transition-none">
        <GlassOrb params={SMALL_ORB_PARAMS}>
          <Icon
            icon={Add01Icon}
            size={18}
            strokeWidth={2.5}
            className="text-white drop-shadow-[0_1px_2px_rgba(76,29,149,0.45)]"
          />
        </GlassOrb>
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
