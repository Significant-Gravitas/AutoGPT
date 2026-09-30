import {
  downloadCopilotSkillPackage,
  getListCopilotSkillsQueryKey,
  readCopilotSkill,
  useDeleteCopilotSkill,
  useReadCopilotSkill,
} from "@/app/api/__generated__/endpoints/skills/skills";
import type { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";
import type { CopilotSkillDetail } from "@/app/api/__generated__/models/copilotSkillDetail";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  buildSkillFileRows,
  describeSkill,
  downloadFile,
  renderSkillMarkdown,
} from "./helpers";

interface Args {
  skill: CopilotSkillInfo;
}

export function useSkillListItem({ skill }: Args) {
  const { toast } = useToast();
  const queryClient = useQueryClient();
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [isViewOpen, setIsViewOpen] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);

  const { mutateAsync: deleteSkill, isPending: isDeleting } =
    useDeleteCopilotSkill();

  const { descriptionPreview, triggers } = describeSkill(skill);

  const {
    data: detailRes,
    isLoading: isDetailLoading,
    error: detailError,
  } = useReadCopilotSkill(skill.name, undefined, {
    query: {
      enabled: isViewOpen,
      staleTime: 60_000,
    },
  });
  const detail =
    detailRes && detailRes.status === 200
      ? (detailRes.data as CopilotSkillDetail)
      : null;
  // Surface either a transport error (network down) or a non-200 response
  // (404 / 500) so the View dialog can show "fetch failed" instead of
  // ambiguously rendering "(no body)".
  const detailErrorMessage = (() => {
    if (detailError instanceof Error) return detailError.message;
    if (detailRes && detailRes.status !== 200) {
      return `Failed to load skill (HTTP ${detailRes.status})`;
    }
    return null;
  })();

  function openDelete() {
    setIsDeleteOpen(true);
  }

  function closeDelete(open: boolean) {
    setIsDeleteOpen(open);
  }

  function openView() {
    setIsViewOpen(true);
  }

  function closeView(open: boolean) {
    setIsViewOpen(open);
  }

  // A package downloads as the zip the upload endpoint takes back; a skill
  // that is only a SKILL.md stays a .md, which is what most users exported.
  async function handleDownload() {
    setIsDownloading(true);
    try {
      const res = await readCopilotSkill(skill.name);
      if (res.status !== 200) {
        throw new Error(`Failed to download skill (HTTP ${res.status})`);
      }
      const skillDetail = res.data as CopilotSkillDetail;
      if (skillDetail.files?.length) {
        const archive = await downloadCopilotSkillPackage(skill.name);
        if (archive.status !== 200) {
          throw new Error(`Failed to download skill (HTTP ${archive.status})`);
        }
        downloadFile(`${skill.name}.zip`, archive.data);
        return;
      }
      downloadFile(
        `${skill.name}.md`,
        new Blob([renderSkillMarkdown(skillDetail)], {
          type: "text/markdown;charset=utf-8",
        }),
      );
    } catch (error) {
      toast({
        title: "Failed to download skill",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      setIsDownloading(false);
    }
  }

  async function handleDelete() {
    try {
      await deleteSkill({ name: skill.name });
      toast({ title: "Skill deleted" });
      setIsDeleteOpen(false);
      queryClient.invalidateQueries({
        queryKey: getListCopilotSkillsQueryKey(),
      });
    } catch (error) {
      toast({
        title: "Failed to delete skill",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    }
  }

  return {
    descriptionPreview,
    triggers,
    fileRows: buildSkillFileRows(detail?.files),
    isDeleteOpen,
    openDelete,
    closeDelete,
    isDeleting,
    handleDelete,
    isDownloading,
    handleDownload,
    isViewOpen,
    openView,
    closeView,
    isDetailLoading,
    detail,
    detailErrorMessage,
  };
}
