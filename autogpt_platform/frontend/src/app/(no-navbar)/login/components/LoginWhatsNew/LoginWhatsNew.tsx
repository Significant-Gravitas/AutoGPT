import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { CHANGELOG_BASE_URL } from "@/components/molecules/ChangelogPopup/changelog-constants";
import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import { useId } from "react";
import { ChangelogFallback } from "./components/ChangelogFallback";
import { ReleaseList } from "./components/ReleaseList";
import { ReleaseListSkeleton } from "./components/ReleaseListSkeleton";
import { useLoginWhatsNew } from "./useLoginWhatsNew";

export function LoginWhatsNew() {
  const { releases, isLoading } = useLoginWhatsNew();
  const headingID = useId();
  const isUnavailable = !isLoading && releases.length === 0;

  return (
    <section
      aria-labelledby={headingID}
      aria-busy={isLoading}
      className="flex max-w-lg flex-col gap-3"
    >
      <div className="flex items-center justify-between gap-4">
        <Text
          as="h2"
          id={headingID}
          variant="small-medium"
          tone="secondary"
          // TODO(tokens): no exact text token for tracking-[0.14em]
          className="uppercase tracking-[0.14em]"
        >
          What&apos;s new
        </Text>
        {isUnavailable ? null : (
          <Link
            href={CHANGELOG_BASE_URL}
            isExternal
            className="inline-flex items-center gap-1 text-caption leading-4 text-foreground-secondary hover:text-foreground-strong"
          >
            Full changelog
            <span className="sr-only">(opens in a new tab)</span>
            <Icon icon={ArrowUpRight01Icon} size={14} aria-hidden />
          </Link>
        )}
      </div>
      <WhatsNewBody isLoading={isLoading} releases={releases} />
    </section>
  );
}

function WhatsNewBody({
  isLoading,
  releases,
}: ReturnType<typeof useLoginWhatsNew>) {
  if (isLoading) return <ReleaseListSkeleton />;
  if (releases.length === 0) return <ChangelogFallback />;
  return <ReleaseList releases={releases} />;
}
