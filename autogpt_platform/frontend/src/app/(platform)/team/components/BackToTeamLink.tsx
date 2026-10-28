import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";

export function BackToTeamLink() {
  return (
    <Link
      href="/team"
      className="group inline-flex items-center gap-1 text-foreground-tertiary hover:text-foreground-emphasis"
      data-testid="expert-back-to-team"
    >
      <Icon icon={ArrowLeft02Icon} size={14} />
      <Text
        variant="body"
        as="span"
        tone="muted"
        className="group-hover:text-foreground-emphasis"
      >
        Back to Team
      </Text>
    </Link>
  );
}
