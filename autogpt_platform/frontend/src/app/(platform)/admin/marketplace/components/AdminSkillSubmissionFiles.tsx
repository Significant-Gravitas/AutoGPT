"use client";

import type { SkillPackageFile } from "@/app/api/__generated__/models/skillPackageFile";
import { Button } from "@/components/atoms/Button/Button";
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
      <Button
        type="button"
        variant="toggle"
        size="xs"
        onClick={() => setIsExpanded(!isExpanded)}
        aria-expanded={isExpanded}
        data-testid={`files-${versionId}`}
        leadingIcon={isExpanded ? ArrowDown01Icon : ArrowRight01Icon}
      >
        {files.length === 1 ? "1 file" : `${files.length} files`}
      </Button>

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
