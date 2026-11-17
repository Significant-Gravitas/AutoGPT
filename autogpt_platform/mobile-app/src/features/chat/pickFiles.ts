import * as DocumentPicker from "expo-document-picker";
import * as ImagePicker from "expo-image-picker";
import { ActionSheetIOS, Alert, Platform } from "react-native";

import { nameFromUri, type PickedFile } from "./attachmentHelpers";

export type AttachSource = "photos" | "files";

/** The native choice between the photo library and files; null when cancelled. */
export function chooseAttachSource() {
  return new Promise<AttachSource | null>((resolve) => {
    if (Platform.OS === "ios") {
      ActionSheetIOS.showActionSheetWithOptions(
        { options: ["Photo library", "Files", "Cancel"], cancelButtonIndex: 2 },
        (index) => resolve(index === 0 ? "photos" : index === 1 ? "files" : null),
      );
      return;
    }
    Alert.alert(
      "Attach",
      undefined,
      [
        { text: "Cancel", style: "cancel", onPress: () => resolve(null) },
        { text: "Files", onPress: () => resolve("files") },
        { text: "Photo library", onPress: () => resolve("photos") },
      ],
      { cancelable: true, onDismiss: () => resolve(null) },
    );
  });
}

export async function pickFrom(source: AttachSource, limit: number) {
  return source === "photos" ? pickPhotos(limit) : pickDocuments();
}

// The system photo picker needs no library permission to choose photos.
async function pickPhotos(limit: number): Promise<PickedFile[]> {
  const result = await ImagePicker.launchImageLibraryAsync({
    mediaTypes: ["images"],
    allowsMultipleSelection: true,
    selectionLimit: limit,
    quality: 1,
  });
  if (result.canceled) return [];
  return result.assets.map((asset) => ({
    name: asset.fileName || nameFromUri(asset.uri, "photo.jpg"),
    uri: asset.uri,
    mimeType: asset.mimeType ?? "image/jpeg",
    size: asset.fileSize,
    isImage: true,
  }));
}

async function pickDocuments(): Promise<PickedFile[]> {
  const result = await DocumentPicker.getDocumentAsync({
    multiple: true,
    copyToCacheDirectory: true,
  });
  if (result.canceled) return [];
  return result.assets.map((asset) => ({
    name: asset.name || nameFromUri(asset.uri, "file"),
    uri: asset.uri,
    mimeType: asset.mimeType ?? "application/octet-stream",
    size: asset.size,
    isImage: asset.mimeType?.startsWith("image/") ?? false,
  }));
}
