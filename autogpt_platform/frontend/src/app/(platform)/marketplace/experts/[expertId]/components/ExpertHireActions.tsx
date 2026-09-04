import { Expert } from "@/app/api/__generated__/models/expert";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";

// Sized and shaped like the small button beside it so the pair reads as one row.
const STATUS_CLASS = "h-9 rounded-full px-3.5 text-sm";

// A white, hairline-bordered secondary so the button sits quietly next to
// the status badge instead of reading as a solid grey slab.
const SECONDARY_CLASS =
  "border-zinc-200 bg-white shadow-[0_1px_2px_rgba(16,24,40,0.05)] hover:border-zinc-300 hover:bg-zinc-50";

interface Props {
  expert: Expert;
  hiredExpert: Expert | null;
  isLoggedIn: boolean;
  isHiring: boolean;
  onHire: () => void;
}

/** The page's one call to action, sitting in the header beside the name.
 *  Signed-out visitors get a sign-up prompt that brings them back here. */
export function ExpertHireActions({
  expert,
  hiredExpert,
  isLoggedIn,
  isHiring,
  onHire,
}: Props) {
  if (!isLoggedIn) {
    const next = encodeURIComponent(`/marketplace/experts/${expert.id}`);
    return (
      <Button
        as="NextLink"
        href={`/signup?next=${next}`}
        variant="primary"
        size="small"
        className="w-full sm:w-auto"
      >
        Get started
      </Button>
    );
  }

  if (hiredExpert) {
    return (
      <div className="flex flex-wrap items-center gap-3">
        <Badge variant="success" className={STATUS_CLASS}>
          <Icon icon={CheckmarkCircle02Icon} size={16} />
          On your team
        </Badge>
        <Button
          as="NextLink"
          href={`/copilot?expertId=${hiredExpert.id}`}
          variant="secondary"
          size="small"
          className={SECONDARY_CLASS}
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
