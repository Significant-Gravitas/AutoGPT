import { getExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  role: string;
  label: string;
}

export function ExpertAreaChip({ role, label }: Props) {
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
      <span className="truncate">{label}</span>
    </Badge>
  );
}
