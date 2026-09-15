import type { PackagedSkillInfo } from "@/app/api/__generated__/models/packagedSkillInfo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface Props {
  skill: PackagedSkillInfo;
  isRemoved: boolean;
  onToggle: () => void;
  readOnly?: boolean;
}

export function SkillRow({ skill, isRemoved, onToggle, readOnly }: Props) {
  return (
    <li
      className={cn(
        "flex items-center justify-between gap-3 rounded-lg px-3 py-2 ring-1 ring-inset ring-zinc-200",
        isRemoved && "opacity-50",
      )}
    >
      <div className="min-w-0">
        <Text variant="body-medium" className="truncate text-zinc-900">
          {skill.name}
        </Text>
        <Text variant="small" className="text-zinc-500">
          {skill.files?.length === 1
            ? "1 file"
            : `${skill.files?.length ?? 0} files`}
        </Text>
      </div>
      {readOnly ? null : (
        <Button
          variant="ghost"
          size="xs"
          onClick={onToggle}
          aria-label={`${isRemoved ? "Keep" : "Remove"} ${skill.name}`}
        >
          {isRemoved ? "Undo" : "Remove"}
        </Button>
      )}
    </li>
  );
}
