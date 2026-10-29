import {
  AiBrain01Icon,
  ArrowRight02Icon,
  LockIcon,
} from "@hugeicons/core-free-icons";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  label: string;
  name: string;
  model?: string | null;
  reason: string;
  href: string;
}

export function MaxUpgradeCard({ label, name, model, reason, href }: Props) {
  return (
    <div className="mt-3 rounded-xl border border-brand/20 bg-accent-surface p-4">
      <div
        role="radio"
        aria-checked={false}
        aria-disabled="true"
        aria-label={`${label} — ${reason}`}
        tabIndex={-1}
        className="flex items-start gap-3"
      >
        <span className="flex size-8 flex-none items-center justify-center rounded-lg border border-brand/30 bg-surface text-brand-hover">
          <Icon icon={AiBrain01Icon} size={18} aria-hidden />
        </span>
        <span className="min-w-0 flex-1">
          <Text
            variant="body-medium-tight"
            as="span"
            tone="primary"
            className="block"
          >
            {name}
          </Text>
          {model && (
            <Text
              variant="small-tight"
              as="span"
              tone="secondary"
              className="mt-0.5 block break-words"
            >
              {model}
            </Text>
          )}
        </span>
        <span className="inline-flex items-center gap-1 rounded bg-surface px-1.5 py-1 text-tiny font-semibold leading-inherit tracking-wide text-accent-strong">
          <Icon icon={LockIcon} size={10} aria-hidden />
          MAX
        </span>
      </div>
      <p className="mb-2 mt-5 text-lg font-medium leading-snug tracking-tight text-foreground-strong">
        Unlock {name} with Max.
      </p>
      <Text variant="small-relaxed" tone="secondary">
        {reason}
      </Text>
      <Button
        as="NextLink"
        href={href}
        variant="primary"
        size="small"
        // oxlint-disable-next-line design-system/no-raw-colors -- the hover is a step darker than brand-hover, and no role is
        className="mt-5 h-11 w-full rounded-lg border-brand bg-brand-hover text-action-foreground hover:border-purple-700 hover:bg-purple-700"
      >
        Upgrade to Max
        <Icon icon={ArrowRight02Icon} size={16} aria-hidden />
      </Button>
    </div>
  );
}
