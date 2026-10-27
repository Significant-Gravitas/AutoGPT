import { Icon } from "@/components/atoms/Icon/Icon";
import { BubbleChatIcon, Coins01Icon } from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

function StageTile({ icon, label }: { icon: IconSvgElement; label: string }) {
  return (
    <div className="flex h-24 w-28 flex-col items-center justify-center gap-2 rounded-2xl bg-surface px-3 shadow-md">
      <Icon
        icon={icon}
        size={24}
        // TODO(tokens): no exact role for text-violet-600
        className="text-violet-600"
      />
      <span className="text-center text-caption font-medium leading-4 text-foreground-strong">
        {label}
      </span>
    </div>
  );
}

export function CreditsStage() {
  return (
    <div className="flex h-full items-center justify-center gap-3" aria-hidden>
      <StageTile icon={BubbleChatIcon} label="Chat usage" />
      <StageTile icon={Coins01Icon} label="Automation credits" />
    </div>
  );
}
