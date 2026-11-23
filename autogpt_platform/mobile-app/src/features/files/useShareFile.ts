import { File, Paths } from "expo-file-system";
import * as Sharing from "expo-sharing";
import { useRef, useState } from "react";
import { Alert } from "react-native";

import { authHeaders } from "@/lib/api/client";
import { fileDownloadURL, type WorkspaceFile } from "@/lib/api/workspace";

/**
 * Downloads a file into the cache and hands it to the phone's share sheet,
 * which is also where "Open in" another app lives.
 */
export function useShareFile() {
  const [sharingID, setSharingID] = useState<string | null>(null);
  // State lands a render late; the ref stops a second tap in the same frame.
  const isBusy = useRef(false);

  async function share(file: WorkspaceFile) {
    if (isBusy.current) return;
    isBusy.current = true;
    setSharingID(file.id);
    try {
      const local = await cachedCopy(file);
      await Sharing.shareAsync(local.uri, {
        mimeType: file.mime_type || undefined,
        dialogTitle: file.name,
      });
    } catch {
      Alert.alert("Couldn't share this file", "Check your connection and try again.");
    } finally {
      isBusy.current = false;
      setSharingID(null);
    }
  }

  return { share, sharingID };
}

async function cachedCopy(file: WorkspaceFile) {
  // The id keeps two files of the same name apart; the name is what the other app shows.
  const target = new File(Paths.cache, `${file.id}-${file.name.replace(/[/\\]/g, "_")}`);
  if (target.exists) return target;
  return File.downloadFileAsync(fileDownloadURL(file.id), target, {
    headers: await authHeaders(),
  });
}
