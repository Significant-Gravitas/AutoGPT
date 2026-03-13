import { uploadFileDirect } from "@/lib/direct-upload";
import type { FileUIPart } from "ai";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useEffect, useRef, useState, type MutableRefObject } from "react";

const MAX_FILES = 10;
const MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024;

interface UploadedFile {
  file_id: string;
  name: string;
  mime_type: string;
}

interface SendMessageInput {
  text: string;
  files?: FileUIPart[];
}

interface Props {
  createSession: () => Promise<unknown>;
  isUserStoppingRef: MutableRefObject<boolean>;
  sendMessage: (input: SendMessageInput) => void;
  sessionId: string | null;
}

async function uploadFiles(
  files: File[],
  sessionId: string,
): Promise<UploadedFile[]> {
  const results = await Promise.allSettled(
    files.map(async (file) => {
      try {
        const data = await uploadFileDirect(file, sessionId);
        if (!data.file_id) throw new Error("No file_id returned");
        return {
          file_id: data.file_id,
          name: data.name || file.name,
          mime_type: data.mime_type || "application/octet-stream",
        } satisfies UploadedFile;
      } catch (error) {
        console.error("File upload failed:", error);
        toast({
          title: "File upload failed",
          description: file.name,
          variant: "destructive",
        });
        throw error;
      }
    }),
  );

  return results
    .filter(
      (result): result is PromiseFulfilledResult<UploadedFile> =>
        result.status === "fulfilled",
    )
    .map((result) => result.value);
}

function buildFileParts(uploaded: UploadedFile[]): FileUIPart[] {
  return uploaded.map((file) => ({
    type: "file" as const,
    mediaType: file.mime_type,
    filename: file.name,
    url: `/api/proxy/api/workspace/files/${file.file_id}/download`,
  }));
}

export function useFileUpload({
  createSession,
  isUserStoppingRef,
  sendMessage,
  sessionId,
}: Props) {
  const [isUploadingFiles, setIsUploadingFiles] = useState(false);
  const [pendingMessage, setPendingMessage] = useState<string | null>(null);
  const pendingFilesRef = useRef<File[]>([]);

  useEffect(() => {
    if (!sessionId || pendingMessage === null) {
      return;
    }

    const message = pendingMessage;
    const files = pendingFilesRef.current;
    setPendingMessage(null);
    pendingFilesRef.current = [];

    if (files.length === 0) {
      sendMessage({ text: message });
      return;
    }

    setIsUploadingFiles(true);
    void uploadFiles(files, sessionId)
      .then((uploaded) => {
        if (uploaded.length === 0) {
          toast({
            title: "File upload failed",
            description: "Could not upload any files. Please try again.",
            variant: "destructive",
          });
          return;
        }

        const fileParts = buildFileParts(uploaded);
        sendMessage({
          text: message,
          files: fileParts.length > 0 ? fileParts : undefined,
        });
      })
      .finally(() => setIsUploadingFiles(false));
  }, [pendingMessage, sendMessage, sessionId]);

  async function onSend(message: string, files?: File[]) {
    const trimmed = message.trim();
    if (!trimmed && (!files || files.length === 0)) {
      return;
    }

    if (files && files.length > 0) {
      if (files.length > MAX_FILES) {
        toast({
          title: "Too many files",
          description: `You can attach up to ${MAX_FILES} files at once.`,
          variant: "destructive",
        });
        return;
      }

      const oversized = files.filter((file) => file.size > MAX_FILE_SIZE_BYTES);
      if (oversized.length > 0) {
        toast({
          title: "File too large",
          description: `${oversized[0].name} exceeds the 100 MB limit.`,
          variant: "destructive",
        });
        return;
      }
    }

    isUserStoppingRef.current = false;

    if (sessionId) {
      if (!files || files.length === 0) {
        sendMessage({ text: trimmed });
        return;
      }

      setIsUploadingFiles(true);
      try {
        const uploaded = await uploadFiles(files, sessionId);
        if (uploaded.length === 0) {
          throw new Error("All file uploads failed");
        }

        const fileParts = buildFileParts(uploaded);
        sendMessage({
          text: trimmed || "",
          files: fileParts.length > 0 ? fileParts : undefined,
        });
      } finally {
        setIsUploadingFiles(false);
      }
      return;
    }

    setPendingMessage(trimmed || "");
    pendingFilesRef.current = files ?? [];
    await createSession();
  }

  return {
    isUploadingFiles,
    onSend,
  };
}
