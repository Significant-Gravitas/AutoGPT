import { useToast } from "@/components/molecules/Toast/use-toast";
import { zodResolver } from "@hookform/resolvers/zod";
import { useState } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { importCompetitorFormSchema } from "./LibraryImportCompetitorDialog";

export function useLibraryImportCompetitorDialog() {
  const [isOpen, setIsOpen] = useState(false);
  const { toast } = useToast();
  const [isConverting, setIsConverting] = useState(false);
  const [importMode, setImportMode] = useState<"file" | "url">("file");

  const form = useForm<z.infer<typeof importCompetitorFormSchema>>({
    resolver: zodResolver(importCompetitorFormSchema),
    defaultValues: {
      workflowFile: "",
      templateUrl: "",
    },
  });

  const onSubmit = async (
    values: z.infer<typeof importCompetitorFormSchema>,
  ) => {
    setIsConverting(true);

    try {
      let body: Record<string, unknown>;

      if (importMode === "url" && values.templateUrl) {
        body = { template_url: values.templateUrl, save: true };
      } else if (importMode === "file" && values.workflowFile) {
        // Decode base64 file to JSON
        const base64Match = values.workflowFile.match(
          /^data:[^;]+;base64,(.+)$/,
        );
        if (!base64Match) {
          throw new Error("Invalid file format");
        }
        const jsonString = atob(base64Match[1]);
        const workflowJson = JSON.parse(jsonString);
        body = { workflow_json: workflowJson, save: true };
      } else {
        throw new Error("Please provide a workflow file or template URL");
      }

      const response = await fetch("/api/import/competitor-workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(
          errorData?.detail || `Import failed (${response.status})`,
        );
      }

      const data = await response.json();

      setIsOpen(false);
      form.reset();

      const notes = data.conversion_notes || [];
      const hasWarnings = notes.some(
        (n: string) => n.includes("warning") || n.includes("Warning"),
      );

      toast({
        title: "Workflow Imported",
        description: hasWarnings
          ? `Imported from ${data.source_format} with warnings. Check the builder for details.`
          : `Successfully imported "${data.source_name}" from ${data.source_format}`,
        variant: hasWarnings ? "default" : "default",
      });

      if (data.graph_id) {
        window.location.href = `/build?flowID=${data.graph_id}`;
      }
    } catch (error) {
      console.error("Import failed:", error);
      toast({
        title: "Import Failed",
        description:
          error instanceof Error
            ? error.message
            : "Failed to import workflow. Please check the file format.",
        variant: "destructive",
        duration: 5000,
      });
    } finally {
      setIsConverting(false);
    }
  };

  return {
    onSubmit,
    isConverting,
    isOpen,
    setIsOpen,
    form,
    importMode,
    setImportMode,
  };
}
