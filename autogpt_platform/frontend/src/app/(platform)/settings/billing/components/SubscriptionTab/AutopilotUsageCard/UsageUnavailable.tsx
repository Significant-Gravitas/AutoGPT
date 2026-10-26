import { Alert02Icon, RefreshIcon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ENTERPRISE_CONTACT_URL } from "../YourPlanCard/useYourPlanCard";

interface Props {
  onRetry: () => void;
}

export function UsageUnavailable({ onRetry }: Props) {
  return (
    <section
      aria-label="Usage"
      className="flex h-full min-h-80 flex-col rounded-2xl border border-border-base bg-surface p-6"
    >
      <Text variant="large-medium" as="h2">
        Usage
      </Text>
      <div className="my-auto py-8">
        <span className="mb-4 flex size-9 items-center justify-center rounded-xl bg-surface-hover text-foreground-secondary">
          <Icon icon={Alert02Icon} size={18} />
        </span>
        <Text variant="large-medium" className="mb-2">
          We couldn’t load your usage
        </Text>
        <Text variant="small" tone="secondary" className="leading-6">
          Your plan and your work aren’t affected. Try again in a moment.
        </Text>
        <div className="mt-5 flex flex-wrap items-center gap-3">
          <Button
            variant="secondary"
            size="small"
            onClick={onRetry}
            leftIcon={<Icon icon={RefreshIcon} size={14} aria-hidden="true" />}
          >
            Try again
          </Button>
          <Button
            as="NextLink"
            href={ENTERPRISE_CONTACT_URL}
            variant="ghost"
            size="small"
          >
            Contact support
          </Button>
        </div>
      </div>
    </section>
  );
}
