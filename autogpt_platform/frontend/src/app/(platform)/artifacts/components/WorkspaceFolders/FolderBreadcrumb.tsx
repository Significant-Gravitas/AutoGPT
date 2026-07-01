"use client";

import { CaretRightIcon, HouseIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";

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
        <HouseIcon size={16} />
        <Text variant="small-medium" as="span">
          Files
        </Text>
      </button>
      <CaretRightIcon size={14} className="text-zinc-400" />
      <Text variant="small-medium" as="span" className="text-zinc-800">
        {folderName}
      </Text>
    </nav>
  );
}
