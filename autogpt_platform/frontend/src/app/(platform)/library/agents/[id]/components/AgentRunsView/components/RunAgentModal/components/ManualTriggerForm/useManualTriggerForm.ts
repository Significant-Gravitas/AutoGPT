import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { useState, useMemo } from "react";

interface UseManualTriggerFormProps {
  agent: GraphMeta;
  onClose: () => void;
}

export function useManualTriggerForm({ agent }: UseManualTriggerFormProps) {
  const [isGenerating, setIsGenerating] = useState(false);

  // Mock API endpoint and key for preview
  const apiEndpoint = useMemo(() => {
    return `https://api.agpt.co/v1/agents/${agent.id}/trigger`;
  }, [agent.id]);

  const apiKey = useMemo(() => {
    return "agpt_" + "x".repeat(32); // Mock API key
  }, []);

  function handleGenerateEndpoint() {
    // This will be implemented when the API is ready
    setIsGenerating(true);
    setTimeout(() => {
      setIsGenerating(false);
    }, 1000);
  }

  return {
    apiEndpoint,
    apiKey,
    isGenerating,
    handleGenerateEndpoint,
  };
}
