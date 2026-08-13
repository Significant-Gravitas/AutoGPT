"use client";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowRight01Icon, Home01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  folderName: string;
  onBack: () => void;
}

export function FolderBreadcrumb({ folderName, onBack }: Props) {
  return (
    <nav
      className="flex items-center gap-1.5 text-zinc-500"
      aria-label="Breadcrumb"
      data-testid="folder-breadcrumb"
    >
      <button
        type="button"
        onClick={onBack}
        className="inline-flex items-center gap-1.5 rounded-md px-1.5 py-1 hover:bg-zinc-100 hover:text-zinc-800"
        data-testid="folder-breadcrumb-root"
      >
        <Icon icon={Home01Icon} size={16} />
        <Text variant="small-medium" as="span">
          Files
        </Text>
      </button>
      <Icon icon={ArrowRight01Icon} size={14} className="text-zinc-400" />
      <Text variant="small-medium" as="span" className="text-zinc-800">
        {folderName}
      </Text>
    </nav>
  );
}
