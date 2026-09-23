"use client";

import type { SkillPackageFile } from "@/app/api/__generated__/models/skillPackageFile";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { SkillFileViewer } from "@/components/contextual/SkillPackage/SkillFileViewer";
import { SkillPackageFileList } from "@/components/contextual/SkillPackage/SkillPackageFileList";
import { ArrowDown01Icon, ArrowRight01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";

interface Props {
  versionId: string;
  files: SkillPackageFile[];
}

export function AdminSkillSubmissionFiles({ versionId, files }: Props) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [openFilePath, setOpenFilePath] = useState<string | null>(null);

  if (files.length === 0) return null;

  return (
    <div className="w-full">
      <button
        type="button"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        className="inline-flex items-center gap-1 rounded-md text-zinc-500 transition-colors hover:text-zinc-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-300 focus-visible:ring-offset-2"
        data-testid={`files-${versionId}`}
      >
        <Icon
          icon={isExpanded ? ArrowDown01Icon : ArrowRight01Icon}
          size={14}
          aria-hidden
        />
        <Text variant="small" className="!text-inherit">
          {files.length === 1 ? "1 file" : `${files.length} files`}
        </Text>
      </button>

      {isExpanded ? (
        <div className="mt-2">
          <SkillPackageFileList files={files} onOpenFile={setOpenFilePath} />
        </div>
      ) : null}

      <SkillFileViewer
        source={{ kind: "submission", versionId }}
        path={openFilePath}
        onClose={() => setOpenFilePath(null)}
      />
    </div>
  );
}
