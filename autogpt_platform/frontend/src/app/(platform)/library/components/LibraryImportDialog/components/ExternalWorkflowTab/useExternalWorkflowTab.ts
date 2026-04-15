import { useToast } from "@/components/molecules/Toast/use-toast";
import { uploadFileDirect } from "@/lib/direct-upload";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { fetchWorkflowFromUrl } from "./fetchWorkflowFromUrl";

function decodeBase64Json(dataUrl: string): string {
  const match = dataUrl.match(/^data:[^;]+;base64,(.+)$/);
  if (!match) throw new Error("Could not read the uploaded file.");
  const binary = atob(match[1]);
  const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
  const json = new TextDecoder().decode(bytes);
  JSON.parse(json); // validate — throws SyntaxError if invalid
  return json;
}

async function uploadJsonAsFile(
  jsonString: string,
): Promise<{ fileId: string; fileName: string; mimeType: string }> {
  const file = new File(
    [new Blob([jsonString], { type: "application/json" })],
    `workflow-${crypto.randomUUID()}.json`,
    { type: "application/json" },
  );
  const uploaded = await uploadFileDirect(file);
  return {
    fileId: uploaded.file_id,
    fileName: uploaded.name,
    mimeType: uploaded.mime_type,
  };
}

function storeAndRedirect(
  fileInfo: { fileId: string; fileName: string; mimeType: string },
  router: ReturnType<typeof useRouter>,
) {
  sessionStorage.setItem(
    "importWorkflowPrompt",
    "Import this workflow and recreate it as an AutoGPT agent",
  );
  sessionStorage.setItem("importWorkflowFile", JSON.stringify(fileInfo));
  router.push("/copilot?source=import&autosubmit=true");
}

export function useExternalWorkflowTab() {
  const { toast } = useToast();
  const router = useRouter();
  const [fileValue, setFileValue] = useState("");
  const [urlValue, setUrlValue] = useState("");
  const [submittingMode, setSubmittingMode] = useState<"url" | "file" | null>(
    null,
  );
  const isSubmitting = submittingMode !== null;

  async function submitWithMode(mode: "url" | "file") {
    setSubmittingMode(mode);
    try {
      const jsonString = await resolveJson(mode);
      if (!jsonString) return;
      storeAndRedirect(await uploadJsonAsFile(jsonString), router);
    } catch (err) {
      toast({
        title: "Upload failed",
        description:
          err instanceof Error ? err.message : "Could not upload the file.",
        variant: "destructive",
      });
    } finally {
      setSubmittingMode(null);
    }
  }

  async function resolveJson(mode: "url" | "file"): Promise<string | null> {
    if (mode === "url") {
      const result = await fetchWorkflowFromUrl(urlValue);
      if (!result.ok) {
        toast({
          title: "Could not fetch workflow",
          description: result.error,
          variant: "destructive",
        });
        return null;
      }
      setUrlValue("");
      return result.json;
    }

    try {
      const json = decodeBase64Json(fileValue);
      setFileValue("");
      return json;
    } catch (err) {
      const isParseError = err instanceof SyntaxError;
      toast({
        title: isParseError ? "Invalid JSON" : "Invalid file",
        description: isParseError
          ? "The uploaded file is not valid JSON."
          : "Could not read the uploaded file.",
        variant: "destructive",
      });
      return null;
    }
  }

  return {
    submitWithMode,
    fileValue,
    setFileValue,
    urlValue,
    setUrlValue,
    isSubmitting,
    submittingMode,
  };
}
