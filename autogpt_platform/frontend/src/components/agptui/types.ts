import type { ButtonProps } from "@/components/agptui/Button";
import React from "react";

export type ButtonAction = {
  label: React.ReactNode;
  variant?: ButtonProps["variant"];
  callback: () => void;
};
