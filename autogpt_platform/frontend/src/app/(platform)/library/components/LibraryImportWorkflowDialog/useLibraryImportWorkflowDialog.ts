import { useToast } from "@/components/molecules/Toast/use-toast";
import { uploadFileDirect } from "@/lib/direct-upload";
import { useRouter } from "next/navigation";
import { useState } from "react";

export function useLibraryImportWorkflowDialog() {
  const { toast } = useToast();
  const router = useRouter();
  const [fileValue, setFileValue] = useState("");
  const [urlValue, setUrlValue] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function submitWithMode(mode: "url" | "file") {
    if (mode === "url" && urlValue) {
      const prompt = `Import this workflow and recreate it as an AutoGPT agent: ${urlValue}`;
      setUrlValue("");
      sessionStorage.setItem("importWorkflowPrompt", prompt);
      router.push("/copilot?source=import&autosubmit=true");
      return;
    }

    if (mode === "file" && fileValue) {
      const base64Match = fileValue.match(/^data:[^;]+;base64,(.+)$/);
      if (!base64Match) {
        toast({
          title: "Invalid file",
          description: "Could not read the uploaded file.",
          variant: "destructive",
        });
        return;
      }

      let jsonString: string;
      try {
        jsonString = atob(base64Match[1]);
        JSON.parse(jsonString); // validate JSON before uploading
      } catch {
        toast({
          title: "Invalid JSON",
          description: "The uploaded file is not valid JSON.",
          variant: "destructive",
        });
        return;
      }

      setIsSubmitting(true);
      try {
        const blob = new Blob([jsonString], { type: "application/json" });
        const file = new File([blob], "workflow.json", {
          type: "application/json",
        });
        const uploaded = await uploadFileDirect(file);

        setFileValue("");
        sessionStorage.setItem(
          "importWorkflowPrompt",
          "Import this workflow and recreate it as an AutoGPT agent",
        );
        sessionStorage.setItem(
          "importWorkflowFile",
          JSON.stringify({
            fileId: uploaded.file_id,
            fileName: uploaded.name,
            mimeType: uploaded.mime_type,
          }),
        );
        router.push("/copilot?source=import&autosubmit=true");
      } catch (err) {
        toast({
          title: "Upload failed",
          description:
            err instanceof Error ? err.message : "Could not upload the file.",
          variant: "destructive",
        });
      } finally {
        setIsSubmitting(false);
      }
    }
  }

  return {
    submitWithMode,
    fileValue,
    setFileValue,
    urlValue,
    setUrlValue,
    isSubmitting,
  };
}
