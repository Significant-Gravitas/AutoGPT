import type { Task } from "@/app/api/__generated__/models/task";

const NAME_MAX = 80;

/** A file name from the task's title, safe on every system. */
export function fileNameOf(title: string, extension: string) {
  const name = title
    .replace(/[\\/:*?"<>|\u0000-\u001f]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, NAME_MAX)
    .trim();
  return `${name || "Task result"}.${extension}`;
}

/** The result as a Markdown file: the task's title, then its result. */
export function resultMarkdown(task: Task, text: string) {
  return `# ${task.title}\n\n${text.trim()}\n`;
}

/** Save a link's file to the user's computer under the given name. */
export function saveFile(href: string, name: string) {
  const link = document.createElement("a");
  link.href = href;
  link.download = name;
  link.rel = "noopener";
  document.body.appendChild(link);
  link.click();
  link.remove();
}

/** Save text as a file, through a link to it that is let go after. */
export function saveText(text: string, name: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  saveFile(url, name);
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
