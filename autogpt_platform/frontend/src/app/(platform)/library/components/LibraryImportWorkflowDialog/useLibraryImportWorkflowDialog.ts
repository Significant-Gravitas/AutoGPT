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

  function submitWithMode(mode: "url" | "file") {
    let prompt: string;

    if (mode === "url" && urlValue) {
      prompt = `Import this workflow and recreate it as an AutoGPT agent: ${urlValue}`;
    } else if (mode === "file" && fileValue) {
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

    sessionStorage.setItem("importWorkflowPrompt", prompt);
    router.push("/copilot?source=import&autosubmit=true");
  }

  function onSubmit() {
    submitWithMode(importMode);
  }

  return {
    onSubmit,
    submitWithMode,
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
