import { useState } from "react";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { usePostV2GenerateTestData } from "@/app/api/__generated__/endpoints/admin/admin";
import type { GenerateTestDataResponse } from "@/app/api/__generated__/models/generateTestDataResponse";
import type { TestDataScriptType } from "@/app/api/__generated__/models/testDataScriptType";

export function useGenerateTestDataButton() {
  const { toast } = useToast();
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [scriptType, setScriptType] = useState<TestDataScriptType>("e2e");
  const [result, setResult] = useState<GenerateTestDataResponse | null>(null);

  const { mutate, isPending } = usePostV2GenerateTestData({
    mutation: {
      onSuccess: (response) => {
        // The mutation types response.data as a status-discriminated union;
        // onSuccess only fires for 2xx, so narrow to the 200 body.
        if (response.status !== 200) return;
        const data = response.data;
        setResult(data);
        toast({
          title: data.success ? "Success" : "Error",
          description: data.message,
          variant: data.success ? undefined : "destructive",
        });
      },
      onError: (error) => {
        const detail =
          error instanceof Error ? error.message : "Unknown error occurred";
        setResult({
          success: false,
          message: `Failed to generate test data: ${detail}`,
        });
        toast({
          title: "Error",
          description: "Failed to generate test data. Please try again.",
          variant: "destructive",
        });
      },
    },
  });

  function openDialog() {
    setResult(null);
    setIsDialogOpen(true);
  }

  function closeDialog() {
    setIsDialogOpen(false);
  }

  function generate() {
    setResult(null);
    mutate({ data: { script_type: scriptType } });
  }

  return {
    isDialogOpen,
    scriptType,
    setScriptType,
    result,
    isPending,
    openDialog,
    closeDialog,
    generate,
  };
}
