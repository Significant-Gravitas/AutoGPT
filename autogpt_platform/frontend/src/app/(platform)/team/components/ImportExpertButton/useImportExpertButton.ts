import {
  importExpertPackage,
  parseExpertPackage,
} from "@/app/api/__generated__/endpoints/experts/experts";
import type { ExpertPackagePreview } from "@/app/api/__generated__/models/expertPackagePreview";
import {
  serializeExpertEdits,
  type ExpertImportEdits,
} from "@/components/contextual/ExpertReviewDialog/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { trackExpertImported } from "@/services/experts/experts-analytics";
import { invalidateExpertRosterQueries } from "@/services/experts/invalidate-experts";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";
import { getImportFailureLine, getPackageFileError } from "./helpers";

export function useImportExpertButton() {
  const { toast } = useToast();
  const router = useRouter();
  const queryClient = useQueryClient();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<ExpertPackagePreview | null>(null);
  const [isParsing, setIsParsing] = useState(false);
  const [isImporting, setIsImporting] = useState(false);

  function openFilePicker() {
    fileInputRef.current?.click();
  }

  function closeDialog() {
    setFile(null);
    setPreview(null);
  }

  function reportFailure(title: string, error: unknown) {
    toast({
      title,
      description:
        error instanceof Error
          ? error.message
          : "An unexpected error occurred.",
      variant: "destructive",
    });
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const picked = event.target.files?.[0];
    // Reset the input so re-picking the same file fires onChange again.
    event.target.value = "";
    if (!picked) return;

    const fileError = getPackageFileError(picked);
    if (fileError) {
      toast({
        title: "Can't import this file",
        description: fileError,
        variant: "destructive",
      });
      return;
    }

    setIsParsing(true);
    try {
      const response = await parseExpertPackage({ file: picked });
      if (response.status !== 200) {
        throw new Error(`Couldn't read this file (HTTP ${response.status})`);
      }
      setFile(picked);
      setPreview(response.data);
    } catch (error) {
      reportFailure("Can't import this file", error);
    } finally {
      setIsParsing(false);
    }
  }

  async function confirmImport(edits: ExpertImportEdits) {
    if (!file) return;

    setIsImporting(true);
    try {
      const response = await importExpertPackage({
        file,
        edits: serializeExpertEdits(edits),
      });
      if (response.status !== 201) {
        throw new Error(
          `Couldn't import this expert (HTTP ${response.status})`,
        );
      }

      const { expert } = response.data;
      toast({
        title: `Imported ${expert.name}`,
        description: getImportFailureLine(response.data) ?? undefined,
      });
      trackExpertImported({
        expert_id: expert.id,
        workflow_count: expert.workflows.length,
        skill_count: expert.skills.length,
      });

      closeDialog();
      await invalidateExpertRosterQueries(queryClient);
      router.push(`/team/${expert.id}`);
    } catch (error) {
      reportFailure("Couldn't import this expert", error);
    } finally {
      setIsImporting(false);
    }
  }

  return {
    fileInputRef,
    isParsing,
    isImporting,
    isDialogOpen: preview !== null,
    preview,
    openFilePicker,
    handleFileChange,
    closeDialog,
    confirmImport,
  };
}
