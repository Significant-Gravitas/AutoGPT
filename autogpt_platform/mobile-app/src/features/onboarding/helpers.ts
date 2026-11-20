import {
  ChartLineData01Icon,
  CodeIcon,
  CubeIcon,
  Flag01Icon,
  Megaphone01Icon,
  Settings02Icon,
  Target01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react-native";

export const OTHER_ROLE = "Other";
export const OTHER_ROLE_MAX_LENGTH = 100;
/** The backend's limit on a typed brain dump. */
export const MAX_TYPED_CHARS = 20_000;
const USER_NAME_MAX_LENGTH = 100;

// The web's role ids (onboarding RoleStep): the backend stores them as sent.
export const ROLES: readonly { id: string; label: string; icon: IconSvgElement }[] = [
  { id: "Founder/CEO", label: "Founder / CEO", icon: Target01Icon },
  { id: "Operations", label: "Operations", icon: Settings02Icon },
  { id: "Sales/BD", label: "Sales / BD", icon: ChartLineData01Icon },
  { id: "Marketing", label: "Marketing", icon: Megaphone01Icon },
  { id: "Product/PM", label: "Product / PM", icon: CubeIcon },
  { id: "Engineering", label: "Engineering", icon: CodeIcon },
  { id: "HR/People", label: "HR / People", icon: UserGroupIcon },
  { id: OTHER_ROLE, label: "Other", icon: Flag01Icon },
];

export type WorkAnswer = { mode: "typed"; text: string } | { mode: "skipped" };

export type OnboardingStepKey = "role" | "work" | "setup" | "plan";

export function canContinueRole(role: string, otherRole: string) {
  if (!role) return false;
  return role !== OTHER_ROLE || otherRole.trim().length > 0;
}

export function canSendTyped(text: string) {
  return text.trim().length > 0 && text.length <= MAX_TYPED_CHARS;
}

/** The role sent to the backend: the pick, or what was written for Other. */
export function resolveRole(role: string, otherRole: string) {
  if (role !== OTHER_ROLE) return role;
  // Code points, as the backend counts them.
  return cutTo(otherRole.trim(), OTHER_ROLE_MAX_LENGTH);
}

/** What Otto calls the person: the first name, or the email's first part. */
export function displayName(user: { name: string; email: string }) {
  const first = user.name.trim().split(/\s+/)[0];
  return cutTo(first || (user.email.split("@")[0] ?? ""), USER_NAME_MAX_LENGTH);
}

function cutTo(text: string, max: number) {
  return Array.from(text).slice(0, max).join("");
}
