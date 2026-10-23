import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { Logout03Icon, MoreVerticalIcon } from "@hugeicons/core-free-icons";

interface Props {
  expertName: string;
  onFire: () => void;
  testId: string;
}

export function FireExpertMenu({ expertName, onFire, testId }: Props) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          type="button"
          variant="secondary"
          size="icon-sm"
          leadingIcon={MoreVerticalIcon}
          aria-label={`${expertName} actions`}
          withTooltip={false}
          data-testid={testId}
          onClick={(event) => event.stopPropagation()}
        />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" sideOffset={6}>
        <DropdownMenuItem
          onSelect={onFire}
          // TODO(tokens): no fitting role for text-red-600 #D93636; #D93636 is error-hover (a hover fill)
          className="flex cursor-pointer items-center gap-2 text-red-600 focus:bg-error-surface focus:text-error-foreground"
        >
          <Icon icon={Logout03Icon} size={14} />
          Fire {expertName}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
