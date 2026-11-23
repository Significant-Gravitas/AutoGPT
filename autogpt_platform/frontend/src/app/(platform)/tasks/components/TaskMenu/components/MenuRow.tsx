import { Icon } from "@/components/atoms/Icon/Icon";
import { DropdownMenuItem } from "@/components/molecules/DropdownMenu/DropdownMenu";
import type { IconSvgElement } from "@hugeicons/react";

interface Props {
  icon: IconSvgElement;
  label: string;
  onSelect: () => void;
}

/** One action in the task menu: its icon and words. */
export function MenuRow({ icon, label, onSelect }: Props) {
  return (
    <DropdownMenuItem onSelect={onSelect}>
      <span className="flex items-center gap-2">
        <Icon icon={icon} size={14} />
        {label}
      </span>
    </DropdownMenuItem>
  );
}
