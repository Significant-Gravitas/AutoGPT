"use client";

import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { getGetV1ListExecutionSchedulesForAGraphQueryKey } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { useToast } from "@/components/molecules/Toast/use-toast";

function getAgentInputFields(agent: LibraryAgent): Record<string, any> {
  const schema = agent.input_schema as unknown as {
    properties?: Record<string, any>;
  } | null;
  if (!schema || !schema.properties) return {};
  const properties = schema.properties as Record<string, any>;
  const visibleEntries = Object.entries(properties).filter(
    ([, sub]) => !sub?.hidden,
  );
  return Object.fromEntries(visibleEntries);
}

export function useEditInputsModal(
  agent: LibraryAgent,
  schedule: GraphExecutionJobInfo,
) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [isOpen, setIsOpen] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const inputFields = useMemo(() => getAgentInputFields(agent), [agent]);
  const [values, setValues] = useState<Record<string, any>>({
    ...(schedule.input_data as Record<string, any>),
  });

  async function handleSave() {
    setIsSaving(true);
    try {
      const res = await fetch(`/api/schedules/${schedule.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ inputs: values }),
      });
      if (!res.ok) {
        let message = "Failed to update schedule inputs";
        const data = await res.json();
        message = data?.message || data?.detail || message;
        throw new Error(message);
      }

      await queryClient.invalidateQueries({
        queryKey: getGetV1ListExecutionSchedulesForAGraphQueryKey(
          schedule.graph_id,
        ),
      });
      toast({
        title: "Schedule inputs updated",
      });
      setIsOpen(false);
    } catch (error: any) {
      toast({
        title: "Failed to update schedule inputs",
        description: error?.message || "An unexpected error occurred.",
        variant: "destructive",
      });
    }
    setIsSaving(false);
  }

  return {
    isOpen,
    setIsOpen,
    inputFields,
    values,
    setValues,
    handleSave,
    isSaving,
  } as const;
}
