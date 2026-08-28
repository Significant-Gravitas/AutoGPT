import BackendAPI from "@/lib/autogpt-server-api";
import { useState } from "react";

interface RunAgentInputScope {
  organizationId: string | null;
  teamId: string | null;
}

export function useRunAgentInputs(scope?: RunAgentInputScope) {
  const baseApi = new BackendAPI();
  const api = scope ? baseApi.withTenantScope(scope) : baseApi;
  const [uploadProgress, setUploadProgress] = useState(0);

  async function handleUploadFile(file: File) {
    const result = await api.uploadFile(file, 24, (progress) =>
      setUploadProgress(progress),
    );
    return result;
  }

  return {
    uploadProgress,
    handleUploadFile,
  };
}
