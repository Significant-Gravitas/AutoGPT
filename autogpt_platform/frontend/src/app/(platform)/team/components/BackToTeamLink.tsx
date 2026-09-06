import { Icon } from "@/components/atoms/Icon/Icon";
import { ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";

export function BackToTeamLink() {
  return (
    <Link
      href="/team"
      className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-800"
      data-testid="expert-back-to-team"
    >
      <Icon icon={ArrowLeft02Icon} size={14} />
      Back to Team
    </Link>
  );
}
