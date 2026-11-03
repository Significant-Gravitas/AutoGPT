import { Icon } from "@/components/atoms/Icon/Icon";
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
        <h2
          id={headingID}
          className="font-sans text-caption font-medium tracking-caps text-foreground-secondary uppercase"
        >
          What&apos;s new
        </h2>
        {isUnavailable ? null : (
          <a
            href={CHANGELOG_BASE_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 rounded-sm font-sans text-caption leading-4 font-medium text-foreground-secondary hover:text-foreground-strong hover:underline focus:ring-2 focus:ring-info focus:ring-offset-2 focus:outline-none"
          >
            Full changelog
            <span className="sr-only">(opens in a new tab)</span>
            <Icon icon={ArrowUpRight01Icon} size={14} aria-hidden />
          </a>
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
