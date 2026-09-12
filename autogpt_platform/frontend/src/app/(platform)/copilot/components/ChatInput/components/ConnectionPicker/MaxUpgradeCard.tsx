import { AiBrain01Icon, ArrowRight02Icon } from "@hugeicons/core-free-icons";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";

interface AdvancedTier {
  label: string;
  name: string;
  model?: string | null;
  reason: string;
}

interface Props {
  advanced?: AdvancedTier;
  chatGPTReason?: string;
  href: string;
}

export function MaxUpgradeCard({ advanced, chatGPTReason, href }: Props) {
  return (
    <div className="mt-3 rounded-xl border border-border bg-muted/70 p-4">
      <p className="mb-4 flex items-center gap-2 text-xs leading-5 text-muted-foreground">
        Included with AutoGPT
        <span className="rounded bg-purple-50 px-1.5 py-0.5 text-[10px] font-semibold tracking-wide text-purple-700">
          MAX
        </span>
      </p>
      <div className="space-y-5">
        {advanced && <AdvancedBenefit advanced={advanced} />}
        {chatGPTReason && <ChatGPTBenefit reason={chatGPTReason} />}
      </div>
      <Button
        as="NextLink"
        href={href}
        variant="primary"
        size="small"
        className="mt-5 h-[42px] w-full rounded-lg border-primary bg-primary text-[13px] text-primary-foreground hover:border-primary/90 hover:bg-primary/90"
        data-fast-goal="subscription_upgrade_intent"
        data-fast-goal-surface="model_picker"
      >
        Upgrade to Max
        <Icon icon={ArrowRight02Icon} size={16} aria-hidden />
      </Button>
    </div>
  );
}

function AdvancedBenefit({ advanced }: { advanced: AdvancedTier }) {
  return (
    <div
      role="radio"
      aria-checked={false}
      aria-disabled="true"
      aria-label={`${advanced.label} — ${advanced.reason}`}
      tabIndex={-1}
      className="flex items-start gap-3"
    >
      <span className="flex h-9 w-8 flex-none items-center justify-center text-foreground">
        <Icon icon={AiBrain01Icon} size={20} aria-hidden />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-medium text-foreground">
          {advanced.name}
        </span>
        {advanced.model && (
          <span className="mt-0.5 block break-words text-xs leading-relaxed text-muted-foreground">
            {advanced.model}
          </span>
        )}
      </span>
    </div>
  );
}

function ChatGPTBenefit({ reason }: { reason: string }) {
  return (
    <div
      role="group"
      aria-label={`Connect ChatGPT — ${reason}`}
      className="flex items-start gap-3"
    >
      <span className="flex h-9 w-8 flex-none items-center justify-center">
        <IntegrationLogo provider="openai" size={20} />
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-medium text-foreground">
          Connect ChatGPT
        </span>
        <span className="mt-0.5 block text-xs leading-relaxed text-muted-foreground">
          <span className="block">Use your existing ChatGPT plan.</span>
          <span className="block">Save AutoGPT credits on chats.</span>
        </span>
      </span>
    </div>
  );
}
