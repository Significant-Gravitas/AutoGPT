export function getFileLabel(filename: string, contentType?: string) {
  if (contentType) {
    const mimeParts = contentType.split("/");
    if (mimeParts.length > 1) {
      return `${mimeParts[1].toUpperCase()} file`;
    }
    return `${contentType} file`;
  }

  const pathParts = filename.split(".");
  if (pathParts.length > 1) {
    const ext = pathParts.pop();
    if (ext) return `${ext.toUpperCase()} file`;
  }
  return "File";
}

export function formatFileSize(bytes: number): string {
  if (bytes >= 1024 * 1024) {
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  } else if (bytes >= 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`;
  } else {
    return `${bytes} B`;
  }
}
