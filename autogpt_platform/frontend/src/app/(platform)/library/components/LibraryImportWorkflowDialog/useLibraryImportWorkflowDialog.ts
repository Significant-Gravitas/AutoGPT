import { usePostV2ImportAWorkflowFromAnotherToolN8nMakeComZapier } from "@/app/api/__generated__/endpoints/import/import";
import type { ImportWorkflowRequest } from "@/app/api/__generated__/models/importWorkflowRequest";
import type { ImportWorkflowResponse } from "@/app/api/__generated__/models/importWorkflowResponse";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useRouter } from "next/navigation";
import { useState } from "react";

export function useLibraryImportWorkflowDialog() {
  const [isOpen, setIsOpen] = useState(false);
  const { toast } = useToast();
  const router = useRouter();
  const [importMode, setImportMode] = useState<"file" | "url">("file");
  const [fileValue, setFileValue] = useState("");
  const [urlValue, setUrlValue] = useState("");

  const { mutateAsync: importWorkflow, isPending: isConverting } =
    usePostV2ImportAWorkflowFromAnotherToolN8nMakeComZapier();

  const hasInput = importMode === "url" ? !!urlValue : !!fileValue;

  async function onSubmit() {
    try {
      let body: ImportWorkflowRequest;

      if (importMode === "url" && urlValue) {
        body = { template_url: urlValue };
      } else if (importMode === "file" && fileValue) {
        const base64Match = fileValue.match(/^data:[^;]+;base64,(.+)$/);
        if (!base64Match) {
          throw new Error("Invalid file format");
        }
        const jsonString = atob(base64Match[1]);
        const workflowJson = JSON.parse(jsonString);
        body = { workflow_json: workflowJson };
      } else {
        throw new Error("Please provide a workflow file or template URL");
      }

      const response = await importWorkflow({ data: body });
      const data = response.data as ImportWorkflowResponse;

      setIsOpen(false);
      setFileValue("");
      setUrlValue("");

      toast({
        title: "Workflow Parsed",
        description: `Detected ${data.source_format} workflow "${data.source_name}". Redirecting to AutoPilot...`,
      });

      // Redirect to AutoPilot with the prompt pre-filled and auto-submitted
      const encodedPrompt = encodeURIComponent(data.copilot_prompt);
      router.push(`/copilot?autosubmit=true#prompt=${encodedPrompt}`);
    } catch (error) {
      console.error("Import failed:", error);
      toast({
        title: "Import Failed",
        description:
          error instanceof Error
            ? error.message
            : "Failed to parse workflow. Please check the file format.",
        variant: "destructive",
        duration: 5000,
      });
    }
  }

  return {
    onSubmit,
    isConverting,
    isOpen,
    setIsOpen,
    importMode,
    setImportMode,
    hasInput,
    fileValue,
    setFileValue,
    urlValue,
    setUrlValue,
  };
}
