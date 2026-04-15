import { postV1UploadFileToCloudStorage } from "@/app/api/__generated__/endpoints/files/files";
import { resolveResponse } from "@/app/api/helpers";
import { useState } from "react";

export function useRunAgentInputs() {
  const [uploadProgress, setUploadProgress] = useState(0);

  async function handleUploadFile(file: File) {
    setUploadProgress(0);
    const result = await resolveResponse(
      postV1UploadFileToCloudStorage({ file }, { expiration_hours: 24 }),
    );
    setUploadProgress(100);
    return result;
  }

  return {
    uploadProgress,
    handleUploadFile,
  };
}
