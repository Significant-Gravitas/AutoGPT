import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { DropdownMenuItem } from "@/components/molecules/DropdownMenu/DropdownMenu";
import { UserAdd01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";

const TEAM_HREF = "/team";

/** Shown while Otto is the only one to pick: where to add an expert. */
export function AddExpertItem() {
  return (
    <DropdownMenuItem asChild>
      <Link href={TEAM_HREF} className="flex-col items-start gap-0.5">
        <span className="flex items-center gap-2">
          <Icon icon={UserAdd01Icon} size={14} />
          Add an expert
        </span>
        <Text as="span" variant="small" tone="faint" className="ml-6">
          Experts on your team can take tasks too.
        </Text>
      </Link>
    </DropdownMenuItem>
  );
}
