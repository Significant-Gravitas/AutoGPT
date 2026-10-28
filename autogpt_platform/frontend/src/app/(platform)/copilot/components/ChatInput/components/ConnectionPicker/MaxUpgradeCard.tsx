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
    <div
      // TODO(tokens): no exact role for border-purple-200/40, bg-purple-50
      className="mt-3 rounded-xl border border-purple-200/40 bg-purple-50 p-4"
    >
      <div
        role="radio"
        aria-checked={false}
        aria-disabled="true"
        aria-label={`${label} — ${reason}`}
        tabIndex={-1}
        className="flex items-start gap-3"
      >
        <span
          // TODO(tokens): no exact role for border-purple-200/60; no fitting role for text-purple-600 #6C2EDF; #6C2EDF is brand-hover (a fill)
          className="flex size-8 flex-none items-center justify-center rounded-lg border border-purple-200/60 bg-surface text-purple-600"
        >
          <Icon icon={AiBrain01Icon} size={18} aria-hidden />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-sm font-medium text-foreground-strong">
            {name}
          </span>
          {model && (
            <span className="mt-0.5 block break-words text-xs text-foreground-secondary">
              {model}
            </span>
          )}
        </span>
        <span
          // TODO(tokens): no exact role for text-purple-700
          className="inline-flex items-center gap-1 rounded bg-surface px-1.5 py-1 text-tiny font-semibold leading-inherit tracking-wide text-purple-700"
        >
          <Icon icon={LockIcon} size={10} aria-hidden />
          MAX
        </span>
      </div>
      <p className="mb-2 mt-5 text-lg font-medium leading-snug tracking-tight text-foreground-strong">
        Unlock {name} with Max.
      </p>
      <p className="text-xs leading-relaxed text-foreground-secondary">
        {reason}
      </p>
      <Button
        as="NextLink"
        href={href}
        variant="primary"
        size="small"
        // TODO(tokens): no exact role for hover:border-purple-700, hover:bg-purple-700; no fitting role for border-purple-600 #6C2EDF; #6C2EDF is brand-hover (a fill)
        className="mt-5 h-11 w-full rounded-lg border-purple-600 bg-brand-hover text-action-foreground hover:border-purple-700 hover:bg-purple-700"
      >
        Upgrade to Max
        <Icon icon={ArrowRight02Icon} size={16} aria-hidden />
      </Button>
    </div>
  );
}
