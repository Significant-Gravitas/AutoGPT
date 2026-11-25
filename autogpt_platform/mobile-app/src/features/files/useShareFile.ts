import { File } from "expo-file-system";
import * as Sharing from "expo-sharing";
import { useRef, useState } from "react";
import { Alert, Platform } from "react-native";

import { authHeaders } from "@/lib/api/client";
import { fileDownloadURL, type WorkspaceFile } from "@/lib/api/workspace";
import { assertSameSession, sessionMark } from "@/lib/auth/token";
import { clearSharedFiles, freshShareTarget } from "@/lib/sharedFiles";

/**
 * Downloads a fresh copy of a file and hands it to the phone's share sheet,
 * which is also where "Open in" another app lives. The copy is deleted after
 * the share (iOS), at the next share, and at sign-out.
 */
export function useShareFile() {
  const [sharingID, setSharingID] = useState<string | null>(null);
  // State lands a render late; the ref stops a second tap in the same frame.
  const isBusy = useRef(false);

  async function share(file: WorkspaceFile) {
    if (isBusy.current) return;
    isBusy.current = true;
    setSharingID(file.id);
    const mark = sessionMark();
    try {
      const local = await File.downloadFileAsync(
        fileDownloadURL(file.id),
        freshShareTarget(file.id, file.name),
        { headers: await authHeaders() },
      );
      // Signed out while it came down: the copy goes, unshared.
      assertSameSession(mark);
      await Sharing.shareAsync(local.uri, {
        mimeType: file.mime_type || undefined,
        dialogTitle: file.name,
      });
      // iOS has handed the file over by the time the sheet closes; Android
      // may still be reading it, so there it goes at the next share or sign-out.
      if (Platform.OS === "ios") clearSharedFiles();
    } catch {
      clearSharedFiles();
      // After a sign-out there is nothing to tell.
      if (sessionMark() === mark) {
        Alert.alert("Couldn't share this file", "Check your connection and try again.");
      }
    } finally {
      isBusy.current = false;
      setSharingID(null);
    }
  }

  return { share, sharingID };
}
