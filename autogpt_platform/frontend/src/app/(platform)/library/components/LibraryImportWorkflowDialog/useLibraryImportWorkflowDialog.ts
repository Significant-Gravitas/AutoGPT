import { useToast } from "@/components/molecules/Toast/use-toast";
import { uploadFileDirect } from "@/lib/direct-upload";
import { useRouter } from "next/navigation";
import { useState } from "react";

const N8N_URL_RE = /n8n\.io\/workflows\/(\d+)/i;
const N8N_TEMPLATES_API = "https://api.n8n.io/api/templates/workflows";

async function fetchN8nWorkflowJson(url: string): Promise<string> {
  const match = url.match(N8N_URL_RE);
  if (!match) throw new Error("Not a valid n8n workflow URL");

  const res = await fetch(`${N8N_TEMPLATES_API}/${match[1]}`);
  if (!res.ok) throw new Error(`n8n template not found (${res.status})`);

  const data = await res.json();
  // n8n API: { workflow: { workflow: { nodes, connections, ... }, name, ... } }
  const template = data?.workflow ?? data;
  const workflow = template?.workflow ?? template;
  if (!workflow?.nodes) throw new Error("Unexpected n8n API response format");
  if (!workflow.name) workflow.name = template?.name ?? data?.name ?? "";
  return JSON.stringify(workflow);
}

function decodeBase64Json(dataUrl: string): string {
  const match = dataUrl.match(/^data:[^;]+;base64,(.+)$/);
  if (!match) throw new Error("Could not read the uploaded file.");
  const json = atob(match[1]);
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

export function useLibraryImportWorkflowDialog() {
  const { toast } = useToast();
  const router = useRouter();
  const [fileValue, setFileValue] = useState("");
  const [urlValue, setUrlValue] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);

  async function submitWithMode(mode: "url" | "file") {
    setIsSubmitting(true);
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
      setIsSubmitting(false);
    }
  }

  async function resolveJson(mode: "url" | "file"): Promise<string | null> {
    if (mode === "url") {
      try {
        const json = await fetchN8nWorkflowJson(urlValue);
        setUrlValue("");
        return json;
      } catch (err) {
        toast({
          title: "Could not fetch workflow",
          description: err instanceof Error ? err.message : "Invalid n8n URL.",
          variant: "destructive",
        });
        return null;
      }
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
  };
}
