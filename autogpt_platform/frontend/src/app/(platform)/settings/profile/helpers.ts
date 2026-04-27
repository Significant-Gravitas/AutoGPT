import type { ProfileDetails } from "@/app/api/__generated__/models/profileDetails";

export const MAX_LINKS = 5;
export const INITIAL_LINK_SLOTS = 3;
export const MAX_BIO_LENGTH = 280;
export const HANDLE_REGEX = /^[a-zA-Z0-9_-]{2,30}$/;

function padLinks(links: string[]): string[] {
  if (links.length >= INITIAL_LINK_SLOTS) return links;
  return [
    ...links,
    ...Array<string>(INITIAL_LINK_SLOTS - links.length).fill(""),
  ];
}

export type ProfileFormState = {
  name: string;
  username: string;
  description: string;
  avatar_url: string;
  links: string[];
};

export function profileToFormState(profile: ProfileDetails): ProfileFormState {
  return {
    name: profile.name ?? "",
    username: profile.username ?? "",
    description: profile.description ?? "",
    avatar_url: profile.avatar_url ?? "",
    links: padLinks(profile.links ?? []),
  };
}

export function isFormDirty(
  initial: ProfileFormState,
  current: ProfileFormState,
): boolean {
  if (
    initial.name !== current.name ||
    initial.username !== current.username ||
    initial.description !== current.description ||
    initial.avatar_url !== current.avatar_url
  ) {
    return true;
  }
  const a = initial.links.filter(Boolean);
  const b = current.links.filter(Boolean);
  if (a.length !== b.length) return true;
  return a.some((link, idx) => link !== b[idx]);
}

export function validateForm(state: ProfileFormState): {
  valid: boolean;
  errors: Partial<Record<"name" | "username" | "description", string>>;
} {
  const errors: Partial<Record<"name" | "username" | "description", string>> =
    {};

  if (!state.name.trim()) {
    errors.name = "Display name is required";
  } else if (state.name.length > 50) {
    errors.name = "Display name must be under 50 characters";
  }

  if (!state.username.trim()) {
    errors.username = "Handle is required";
  } else if (!HANDLE_REGEX.test(state.username)) {
    errors.username = "Use 2–30 letters, numbers, underscores or dashes";
  }

  if (state.description.length > MAX_BIO_LENGTH) {
    errors.description = `Bio must be under ${MAX_BIO_LENGTH} characters`;
  }

  return { valid: Object.keys(errors).length === 0, errors };
}

export function getInitials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0]!.slice(0, 2).toUpperCase();
  return (parts[0]![0]! + parts[parts.length - 1]![0]!).toUpperCase();
}
