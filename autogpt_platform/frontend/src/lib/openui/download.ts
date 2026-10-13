export function downloadText(filename: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename.replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 120);
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function csvText(rows: unknown[][]) {
  return rows
    .map((row) =>
      row
        .map((value) => {
          const text = value == null ? "" : String(value);
          const safe =
            typeof value === "string" && /^[\s]*[=+\-@\t\r]/.test(text)
              ? `'${text}`
              : text;
          return `"${safe.replaceAll('"', '""')}"`;
        })
        .join(","),
    )
    .join("\r\n");
}
