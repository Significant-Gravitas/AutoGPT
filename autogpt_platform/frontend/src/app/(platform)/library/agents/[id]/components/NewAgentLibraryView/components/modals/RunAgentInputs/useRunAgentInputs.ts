import BackendAPI from "@/lib/autogpt-server-api";
import { useState } from "react";

export function useRunAgentInputs() {
  const api = new BackendAPI();
  const [uploadProgress, setUploadProgress] = useState(0);

  async function handleUploadFile(file: File) {
    const result = await api.uploadFile(file, "gcs", 24, (progress) =>
      setUploadProgress(progress),
    );
    return result;
  }

  return {
    uploadProgress,
    handleUploadFile,
  };
}
