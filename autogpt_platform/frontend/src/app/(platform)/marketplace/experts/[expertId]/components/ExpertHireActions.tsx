import { Expert } from "@/app/api/__generated__/models/expert";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";

interface Props {
  expert: Expert;
  hiredExpert: Expert | null;
  isHiring: boolean;
  onHire: () => void;
}

/** The page's one call to action, sitting in the header beside the name. */
export function ExpertHireActions({
  expert,
  hiredExpert,
  isHiring,
  onHire,
}: Props) {
  if (hiredExpert) {
    return (
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
        <Badge variant="success" className="w-fit">
          <Icon icon={CheckmarkCircle02Icon} size={14} />
          On your team
        </Badge>
        <Button
          as="NextLink"
          href={`/copilot?expertId=${hiredExpert.id}`}
          variant="secondary"
          size="small"
        >
          {`Chat with ${expert.name}`}
        </Button>
      </div>
    );
  }

  return (
    <Button
      variant="primary"
      size="small"
      onClick={onHire}
      loading={isHiring}
      className="w-full sm:w-auto"
    >
      {`Hire ${expert.name}`}
    </Button>
  );
}
