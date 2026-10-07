import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";

export const MENU_LABEL = "Install for an expert";

interface Props {
  experts: Expert[];
  isAdding: boolean;
  onAddToLibrary: () => void;
  onAddToExpert: (expert: Expert) => void;
}

/** Installing to your own skills is the main action; the dropdown hands the
 *  skill to one of your experts instead. */
export function InstallSkillButton({
  experts,
  isAdding,
  onAddToLibrary,
  onAddToExpert,
}: Props) {
  if (experts.length === 0) {
    return (
      <Button
        variant="primary"
        size="small"
        onClick={onAddToLibrary}
        loading={isAdding}
        className="w-full sm:w-auto"
        data-testid="skill-install-button"
      >
        Install skill
      </Button>
    );
  }

  return (
    <div className="inline-flex w-full sm:w-auto">
      <Button
        variant="primary"
        size="small"
        onClick={onAddToLibrary}
        loading={isAdding}
        className="!rounded-r-none border-r-0 sm:w-auto"
        data-testid="skill-install-button"
      >
        Install skill
      </Button>
      <ExpertInstallMenu
        experts={experts}
        isAdding={isAdding}
        onAddToExpert={onAddToExpert}
      >
        <Button
          variant="primary"
          size="small"
          aria-label={MENU_LABEL}
          disabled={isAdding}
          className="!min-w-0 !rounded-l-none border-l border-l-white/25 px-2.5"
        >
          <Icon icon={ArrowDown01Icon} size={16} aria-hidden />
        </Button>
      </ExpertInstallMenu>
    </div>
  );
}

interface MenuProps {
  experts: Expert[];
  isAdding: boolean;
  onAddToExpert: (expert: Expert) => void;
  children: React.ReactNode;
}

export function ExpertInstallMenu({
  experts,
  isAdding,
  onAddToExpert,
  children,
}: MenuProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>{children}</DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="min-w-[15rem]">
        <DropdownMenuLabel className="text-xs font-normal text-zinc-500">
          {MENU_LABEL}
        </DropdownMenuLabel>
        {experts.map((expert) => (
          <DropdownMenuItem
            key={expert.id}
            disabled={isAdding}
            onSelect={() => onAddToExpert(expert)}
            className="gap-2.5 py-2"
          >
            <Avatar className="h-6 w-6">
              {expert.avatar_url ? (
                <AvatarImage src={expert.avatar_url} alt="" />
              ) : null}
              <AvatarFallback>{expert.name}</AvatarFallback>
            </Avatar>
            <span className="min-w-0">
              <span className="block truncate text-zinc-900">
                {expert.name}
              </span>
              {expert.role ? (
                <span className="block truncate text-xs text-zinc-500">
                  {expert.role}
                </span>
              ) : null}
            </span>
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
