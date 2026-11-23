import { useLocalSearchParams } from "expo-router";

import { FilesPage } from "@/features/files/library/FilesPage";

export default function FolderScreen() {
  const { folderId } = useLocalSearchParams<{ folderId: string }>();
  // Keyed: the drawer keeps this screen mounted, and another folder starts fresh.
  return <FilesPage key={folderId} folderID={folderId} />;
}
