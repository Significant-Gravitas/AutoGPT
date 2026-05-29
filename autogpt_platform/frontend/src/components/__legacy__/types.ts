import type { ButtonProps } from "@/components/__legacy__/Button";
import type { LinkProps } from "next/link";
import React from "react";

export type ButtonAction = {
  label: React.ReactNode;
  variant?: ButtonProps["variant"];
  disabled?: boolean;
  extraProps?: Record<`data-${string}`, string>;
} & (
  | {
      callback: () => void;
      extraProps?: Partial<ButtonProps>;
    }
  | {
      href: string;
      extraProps?: Partial<LinkProps>;
    }
);
