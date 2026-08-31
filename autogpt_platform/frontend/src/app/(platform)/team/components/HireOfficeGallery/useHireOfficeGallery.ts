import { getListExpertsQueryKey } from "@/app/api/__generated__/endpoints/experts/experts";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  HireOfficeResponse,
  OfficeTemplate,
  useHireOfficeMutation,
  useOfficeTemplatesQuery,
} from "./api";

export function useHireOfficeGallery() {
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hireResult, setHireResult] = useState<HireOfficeResponse | null>(null);

  const templatesQuery = useOfficeTemplatesQuery();
  const templates = Array.isArray(templatesQuery.data)
    ? templatesQuery.data
    : [];

  const hireMutation = useHireOfficeMutation({
    onSuccess: (result) => {
      setHireResult(result);
      void queryClient.invalidateQueries({
        queryKey: getListExpertsQueryKey(),
      });
    },
    onError: (error) => {
      toast({
        title: "Could not hire this office",
        description: error instanceof Error ? error.message : undefined,
        variant: "destructive",
      });
    },
  });

  function openPreview(templateId: string) {
    setHireResult(null);
    setSelectedId(templateId);
  }

  function closePreview() {
    setSelectedId(null);
    setHireResult(null);
  }

  function hire(template: OfficeTemplate) {
    hireMutation.mutate({ template_id: template.id });
  }

  return {
    templates,
    isLoading: templatesQuery.isLoading,
    isError: templatesQuery.isError,
    refetch: () => templatesQuery.refetch(),
    selectedTemplate:
      templates.find((template) => template.id === selectedId) ?? null,
    openPreview,
    closePreview,
    hire,
    isHiring: hireMutation.isPending,
    hireResult,
  };
}
