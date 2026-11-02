export const MAX_FILE_NAME_LENGTH = 255;

export function validateFileName(name: string): string | null {
  if (name.length === 0) return "Enter a file name";
  if (name.length > MAX_FILE_NAME_LENGTH) return "That name is too long";
  if (name === "." || name === "..") return "That is not a valid file name";
  if (name.includes("/") || name.includes("\\"))
    return "File names cannot contain slashes";
  return null;
}
