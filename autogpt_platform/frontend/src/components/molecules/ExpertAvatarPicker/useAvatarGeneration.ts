import {
  useGenerateExpertAvatar,
  useGetExpertAvatarGeneration,
} from "@/app/api/__generated__/endpoints/experts/experts";
import type { ExpertAvatarRequest } from "@/app/api/__generated__/models/expertAvatarRequest";
import { useState } from "react";

export function useAvatarGeneration() {
  const [jobID, setJobID] = useState("");
  const mutation = useGenerateExpertAvatar<Error>({
    mutation: { retry: false },
  });
  const query = useGetExpertAvatarGeneration(jobID, {
    query: {
      enabled: Boolean(jobID),
      retry: false,
      refetchInterval: (query) => {
        if (query.state.error) return false;
        const response = query.state.data;
        return !response ||
          (response.status === 200 && response.data.status === "pending")
          ? 2000
          : false;
      },
    },
  });
  const job = query.data?.status === 200 ? query.data.data : undefined;
  const isGenerating =
    mutation.isPending ||
    Boolean(jobID && !query.isError && (!job || job.status === "pending"));
  const error =
    mutation.error?.message ??
    (query.isError
      ? "Could not check generation. Try again or choose a catalog avatar."
      : job?.error);

  function generate(request: ExpertAvatarRequest) {
    setJobID("");
    mutation.mutate(
      { data: request },
      {
        onSuccess: (response) => {
          if (response.status === 202 && response.data.id)
            setJobID(response.data.id);
        },
      },
    );
  }

  return { generate, isGenerating, error, job };
}
