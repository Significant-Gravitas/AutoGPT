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

  const hasInput = importMode === "url" ? !!urlValue : !!fileValue;

  function onSubmit() {
    let prompt: string;

    if (importMode === "url" && urlValue) {
      // Just pass the URL to AutoPilot — it has the import_workflow tool
      prompt = `Import this workflow and recreate it as an AutoGPT agent: ${urlValue}`;
    } else if (importMode === "file" && fileValue) {
      // Decode the base64 file and pass the JSON to AutoPilot
      const base64Match = fileValue.match(/^data:[^;]+;base64,(.+)$/);
      if (!base64Match) {
        toast({
          title: "Invalid file",
          description: "Could not read the uploaded file.",
          variant: "destructive",
        });
        return;
      }
      try {
        const jsonString = atob(base64Match[1]);
        // Validate it's valid JSON
        JSON.parse(jsonString);
        prompt = `Import this workflow JSON and recreate it as an AutoGPT agent:\n\`\`\`json\n${jsonString}\n\`\`\``;
      } catch {
        toast({
          title: "Invalid JSON",
          description: "The uploaded file is not valid JSON.",
          variant: "destructive",
        });
        return;
      }
    } else {
      return;
    }

    setIsOpen(false);
    setFileValue("");
    setUrlValue("");

    toast({
      title: "Redirecting to AutoPilot",
      description: "AutoPilot will import and convert the workflow for you.",
    });

    // Use sessionStorage for large prompts to avoid URL length limits
    sessionStorage.setItem("importWorkflowPrompt", prompt);
    router.push("/copilot?source=import&autosubmit=true");
  }

  return {
    onSubmit,
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
