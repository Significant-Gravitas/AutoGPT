import {
  AiBrain01Icon,
  ArrowRight02Icon,
  LockIcon,
} from "@hugeicons/core-free-icons";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  label: string;
  name: string;
  model?: string | null;
  reason: string;
  href: string;
}

export function MaxUpgradeCard({ label, name, model, reason, href }: Props) {
  return (
    <div className="mt-3 rounded-xl border border-purple-200/40 bg-purple-50 p-4">
      <div
        role="radio"
        aria-checked={false}
        aria-disabled="true"
        aria-label={`${label} — ${reason}`}
        tabIndex={-1}
        className="flex items-start gap-3"
      >
        <span className="flex size-8 flex-none items-center justify-center rounded-lg border border-purple-200/60 bg-white text-purple-600">
          <Icon icon={AiBrain01Icon} size={18} aria-hidden />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium text-zinc-900">
            {name}
          </span>
          {model && (
            <span className="mt-0.5 block break-words text-xs text-zinc-600">
              {model}
            </span>
          )}
        </span>
        <span className="inline-flex items-center gap-1 rounded bg-white px-1.5 py-1 text-[10px] font-semibold tracking-wide text-purple-700">
          <Icon icon={LockIcon} size={10} aria-hidden />
          MAX
        </span>
      </div>
      <p className="mb-2 mt-5 text-lg font-medium leading-snug tracking-tight text-zinc-900">
        Unlock {name} with Max.
      </p>
      <p className="text-xs leading-relaxed text-zinc-600">{reason}</p>
      <Button
        as="NextLink"
        href={href}
        variant="primary"
        size="small"
        className="mt-5 h-11 w-full rounded-lg border-purple-600 bg-purple-600 text-white hover:border-purple-700 hover:bg-purple-700"
        data-fast-goal="subscription_upgrade_intent"
        data-fast-goal-surface="model_picker"
      >
        Upgrade to Max
        <Icon icon={ArrowRight02Icon} size={16} aria-hidden />
      </Button>
    </div>
  );
}
