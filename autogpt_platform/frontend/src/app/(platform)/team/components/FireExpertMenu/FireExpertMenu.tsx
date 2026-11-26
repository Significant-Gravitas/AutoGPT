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
          shape="soft"
          leadingIcon={MoreVerticalIcon}
          aria-label={`${expertName} actions`}
          withTooltip={false}
          data-testid={testId}
          onClick={(event) => event.stopPropagation()}
        />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" sideOffset={6}>
        <DropdownMenuItem onSelect={onFire} tone="error-strong">
          <span className="flex items-center gap-2">
            <Icon icon={Logout03Icon} size={14} />
            Fire {expertName}
          </span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
