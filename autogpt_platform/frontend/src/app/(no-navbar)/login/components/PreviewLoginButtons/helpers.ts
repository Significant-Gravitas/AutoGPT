export const PREVIEW_ROLES = [
  { role: "admin", label: "Admin" },
  { role: "existing", label: "Existing user" },
  { role: "clean", label: "Clean user" },
  { role: "pro", label: "Pro" },
  { role: "enterprise", label: "Enterprise" },
] as const;

export type PreviewRole = (typeof PREVIEW_ROLES)[number]["role"];
