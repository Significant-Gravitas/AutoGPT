export const OPENUI_MIME = "application/vnd.autogpt.openui";
export const OPENUI_EXTENSION = ".openui";

export function isOpenUIFile(mimeType?: string | null, fileName?: string) {
  const mime = (mimeType ?? "").toLowerCase().split(";")[0]?.trim();
  return (
    mime === OPENUI_MIME ||
    (fileName ?? "").toLowerCase().endsWith(OPENUI_EXTENSION)
  );
}

export function defaultViewFileName(title: string) {
  const base = title.replace(/[\\/]/g, "-").trim().slice(0, 200) || "view";
  return `${base}${OPENUI_EXTENSION}`;
}
