import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { CHANGELOG_BASE_URL } from "@/components/molecules/ChangelogPopup/changelog-constants";
import { ArrowUpRight01Icon, NewsIcon } from "@hugeicons/core-free-icons";

// Shown when the changelog can't be read: the section still leads somewhere.
export function ChangelogFallback() {
  return (
    <a
      href={CHANGELOG_BASE_URL}
      target="_blank"
      rel="noopener noreferrer"
      className="group flex items-center gap-4 rounded-2xl bg-surface p-4 transition-colors duration-150 smooth-shadow-ring-sm hover:bg-surface-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-info motion-reduce:transition-none"
    >
      <span
        aria-hidden
        className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-accent-surface text-accent-strong"
      >
        <Icon icon={NewsIcon} size={20} />
      </span>
      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <Text as="span" variant="body-medium" tone="primary">
          See what we&apos;ve shipped lately
        </Text>
        <Text as="span" variant="body" tone="secondary">
          Release notes for Otto, your Experts and the AutoGPT Platform
          <span className="sr-only"> (opens in a new tab)</span>
        </Text>
      </span>
      <Icon
        icon={ArrowUpRight01Icon}
        size={16}
        aria-hidden
        className="shrink-0 text-foreground-tertiary transition-[color,transform] duration-150 group-hover:-translate-y-px group-hover:translate-x-px group-hover:text-foreground-strong motion-reduce:transition-none motion-reduce:group-hover:transform-none"
      />
    </a>
  );
}
