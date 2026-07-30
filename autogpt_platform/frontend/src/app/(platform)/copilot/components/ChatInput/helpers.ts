// Browsers name clipboard screenshots "image.png"; rename so multiple
// pasted images stay distinguishable in the composer and workspace.
const GENERIC_CLIPBOARD_IMAGE_NAME = /^image\.\w+$/i;

export function getFilesFromClipboard(
  clipboardData: DataTransfer | null,
): File[] {
  if (!clipboardData) return [];
  return Array.from(clipboardData.files).map(renameGenericImage);
}

function renameGenericImage(file: File, index: number): File {
  if (
    !file.type.startsWith("image/") ||
    !GENERIC_CLIPBOARD_IMAGE_NAME.test(file.name)
  ) {
    return file;
  }
  const extension = file.name.split(".").pop();
  const stamp = new Date().toISOString().slice(0, 23).replace(/[T:.]/g, "-");
  const suffix = index > 0 ? `-${index + 1}` : "";
  return new File([file], `pasted-image-${stamp}${suffix}.${extension}`, {
    type: file.type,
    lastModified: file.lastModified,
  });
}

export function formatElapsedTime(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  return `${minutes}:${remainingSeconds.toString().padStart(2, "0")}`;
}
