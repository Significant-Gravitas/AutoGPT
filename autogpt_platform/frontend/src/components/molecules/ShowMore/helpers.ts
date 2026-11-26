import { TextVariant } from "@/components/atoms/Text/Text";

export type ShowMoreTextVariant = Exclude<
  TextVariant,
  "h1" | "h2" | "h3" | "h4"
>;

export function getIconSize(variant: ShowMoreTextVariant): number {
  switch (variant) {
    case "lead":
      return 20;
    case "large":
      return 16;
    case "body":
      return 14;
    case "small":
      return 12;
    default:
      return 14;
  }
}
