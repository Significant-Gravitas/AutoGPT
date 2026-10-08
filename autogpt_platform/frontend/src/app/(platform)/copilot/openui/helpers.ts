export async function streamSample(
  source: string,
  signal: AbortSignal,
  update: (source: string) => void,
) {
  for (let end = 0; end < source.length; end += 100) {
    signal.throwIfAborted();
    update(source.slice(0, end + 100));
    await new Promise((resolve) => setTimeout(resolve, 18));
  }
  signal.throwIfAborted();
  return source;
}

export function exportWorkspace(source: string) {
  const url = URL.createObjectURL(
    new Blob([source], { type: "text/plain;charset=utf-8" }),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "autogpt-workspace.openui";
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
