export function downloadText(filename: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename.replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 120);
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
