import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { CHANGELOG_BASE_URL } from "@/components/molecules/ChangelogPopup/changelog-constants";

interface Props {
  children?: string;
  withChangelog?: boolean;
}

export function ThanksLine({
  children = "Thanks, we read every one.",
  withChangelog = true,
}: Props) {
  return (
    <div className="flex items-start gap-2">
      <Icon
        icon={CheckmarkCircle02Icon}
        size={16}
        tone="success-strong"
        className="mt-[3px] shrink-0"
      />
      <Text variant="body" tone="emphasis">
        {children}{" "}
        {withChangelog ? (
          <Link
            href={CHANGELOG_BASE_URL}
            className="text-foreground-tertiary underline underline-offset-2"
          >
            See what we&apos;ve changed lately
          </Link>
        ) : null}
      </Text>
    </div>
  );
}
