import type { ButtonProps } from "@/components/agptui/Button";

export type ButtonAction = {
  label: string;
  variant?: ButtonProps["variant"];
  callback: () => void;
};
