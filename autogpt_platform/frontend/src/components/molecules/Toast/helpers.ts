import type { ToastAction, ToastState } from "@/components/ui/toast";
import { isValidElement, ReactNode } from "react";

export type ToastVariant = "default" | "destructive" | "success" | "info";

export const STATE_BY_VARIANT: Record<ToastVariant, ToastState | undefined> = {
  default: undefined,
  destructive: "error",
  success: "success",
  info: "info",
};

// Longest delay setTimeout accepts; a toast that is not dismissable waits it out.
export const PERSISTENT_LIFETIME = 2_147_483_647;

/** Kobra toasts carry text, so a node collapses to the text it renders. */
export function toText(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") {
    return String(node);
  }
  if (Array.isArray(node)) return node.map(toText).join("");
  if (isValidElement<{ children?: ReactNode }>(node)) {
    return toText(node.props.children);
  }
  return "";
}

interface ActionLike {
  children?: ReactNode;
  onClick?: () => void;
}

/** A button-like action node becomes Kobra's `{ label, run }`. */
export function toAction(node: ReactNode): ToastAction | undefined {
  if (!isValidElement<ActionLike>(node)) return undefined;
  const label = toText(node.props.children);
  const run = node.props.onClick;
  if (!label || typeof run !== "function") return undefined;
  return { label, run: () => run() };
}
