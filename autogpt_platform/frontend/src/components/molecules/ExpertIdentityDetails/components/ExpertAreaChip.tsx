import { getExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Icon } from "@/components/atoms/Icon/Icon";
import { getExpertRoleLabel } from "@/services/experts/expert-role-label";

interface Props {
  role: string;
}

export function ExpertAreaChip({ role }: Props) {
  return (
    <Badge
      variant="info"
      className="max-w-full self-start rounded-full px-2.5 leading-[1.125rem]"
    >
      <Icon
        icon={getExpertAccent(role).roleIcon}
        size={12}
        className="shrink-0"
        aria-hidden
      />
      <span className="truncate">{getExpertRoleLabel(role)}</span>
    </Badge>
  );
}
