import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";

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
        className="mt-[3px] shrink-0 text-success-foreground"
      />
      <Text variant="body" className="!text-foreground-emphasis">
        {children}{" "}
        {withChangelog ? (
          <Link
            href="#changelog"
            className="text-foreground-tertiary underline underline-offset-2"
          >
            See what we&apos;ve changed lately
          </Link>
        ) : null}
      </Text>
    </div>
  );
}
