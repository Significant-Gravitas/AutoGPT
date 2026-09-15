import {
  EXPRESSIONS,
  type ExpressionId,
} from "@/components/molecules/BotAvatar/expressions";
import {
  STATUSES,
  type AvatarStatus,
} from "@/components/molecules/BotAvatar/helpers";

export const PREVIEW_SIZES = [24, 40, 80, 160] as const;
export type PreviewSize = (typeof PREVIEW_SIZES)[number];

export const STATUS_OPTIONS = STATUSES.map((status) => ({
  id: status.id,
  label: status.label,
}));

export const EXPRESSION_OPTIONS = [
  { id: "auto" as const, label: "Follow status" },
  ...EXPRESSIONS.map((expression) => ({
    id: expression.id,
    label: expression.label,
  })),
];

export type ExpressionChoice = "auto" | ExpressionId;

export function isAvatarStatusChoice(value: string): value is AvatarStatus {
  return STATUS_OPTIONS.some((option) => option.id === value);
}

export function downloadNameFor(url: string) {
  return url.split("/").pop() ?? "avatar.svg";
}
