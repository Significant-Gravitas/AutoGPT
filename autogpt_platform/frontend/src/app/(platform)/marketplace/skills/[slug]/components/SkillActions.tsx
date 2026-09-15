import { Expert } from "@/app/api/__generated__/models/expert";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import {
  ArrowDown01Icon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import {
  ExpertInstallMenu,
  InstallSkillButton,
  MENU_LABEL,
} from "./InstallSkillButton";

// Sized and shaped like the small button beside it so the pair reads as one row.
const STATUS_CLASS = "h-9 rounded-full px-3.5 text-sm";

const SECONDARY_CLASS =
  "border-zinc-200 bg-white shadow-[0_1px_2px_rgba(16,24,40,0.05)] hover:border-zinc-300 hover:bg-zinc-50";

interface Props {
  slug: string;
  isLoggedIn: boolean;
  isReady: boolean;
  isAdded: boolean;
  isAdding: boolean;
  experts: Expert[];
  onAdd: () => void;
  onAddToExpert: (expert: Expert) => void;
}

/** The page's one call to action, in the header beside the title. A visitor
 *  gets the same promise, with a return path that makes it true. */
export function SkillActions({
  slug,
  isLoggedIn,
  isReady,
  isAdded,
  isAdding,
  experts,
  onAdd,
  onAddToExpert,
}: Props) {
  return (
    <div aria-live="polite" className="w-full sm:w-auto">
      {!isReady ? (
        <Skeleton className="h-9 w-28 rounded-full" />
      ) : !isLoggedIn ? (
        <Button
          as="NextLink"
          href={`/signup?next=${encodeURIComponent(`/marketplace/skills/${slug}`)}`}
          variant="primary"
          size="small"
          className="w-full sm:w-auto"
        >
          Install skill
        </Button>
      ) : isAdded ? (
        <div className="flex flex-wrap items-center gap-3">
          <Badge variant="success" className={STATUS_CLASS}>
            <Icon icon={CheckmarkCircle02Icon} size={16} aria-hidden />
            Installed
          </Badge>
          <Button
            as="NextLink"
            href="/library/skills"
            variant="secondary"
            size="small"
            className={SECONDARY_CLASS}
          >
            Your skills
          </Button>
          {/* Already in your own skills is no reason not to hand it to an
              expert too, so the dropdown outlives the main action. */}
          {experts.length > 0 ? (
            <ExpertInstallMenu
              experts={experts}
              isAdding={isAdding}
              onAddToExpert={onAddToExpert}
            >
              <Button
                variant="secondary"
                size="small"
                aria-label={MENU_LABEL}
                disabled={isAdding}
                className={`!min-w-0 px-2.5 ${SECONDARY_CLASS}`}
              >
                <Icon icon={ArrowDown01Icon} size={16} aria-hidden />
              </Button>
            </ExpertInstallMenu>
          ) : null}
        </div>
      ) : (
        <InstallSkillButton
          experts={experts}
          isAdding={isAdding}
          onAddToLibrary={onAdd}
          onAddToExpert={onAddToExpert}
        />
      )}
    </div>
  );
}
