import { TextVariant } from "@/components/atoms/Text/Text";

export type ShowMoreTextVariant = Exclude<
  TextVariant,
  "h1" | "h2" | "h3" | "h4"
>;
