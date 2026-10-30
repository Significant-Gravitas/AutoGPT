/**
 * Tutorial Styles for New Builder
 *
 * CSS file contains:
 * - Dynamic classes: .new-builder-tutorial-disable, .new-builder-tutorial-highlight, .new-builder-tutorial-pulse
 * - Shepherd.js overrides
 *
 * Typography (body, small, action, info, tip, warning) uses Tailwind utilities directly in steps.ts
 */
import "shepherd.js/dist/css/shepherd.css";
import "./tutorial.css";

export const injectTutorialStyles = () => {
  if (typeof window !== "undefined") {
    document.documentElement.setAttribute("data-tutorial-styles", "loaded");
  }
};

export const removeTutorialStyles = () => {
  if (typeof window !== "undefined") {
    document.documentElement.removeAttribute("data-tutorial-styles");
  }
};

// Reusable banner components with consistent styling

type BannerVariant = "action" | "info" | "warning" | "success";

const bannerStyles: Record<
  BannerVariant,
  { bg: string; ring: string; text: string }
> = {
  action: {
    bg: "bg-accent-surface",
    ring: "ring-brand/20",
    text: "text-accent-strong",
  },
  info: {
    bg: "bg-info-surface",
    ring: "ring-info/30",
    text: "text-info-foreground",
  },
  warning: {
    bg: "bg-warning-surface",
    ring: "ring-warning-border",
    text: "text-warning-foreground",
  },
  success: {
    bg: "bg-success-surface",
    ring: "ring-success/40",
    text: "text-success-foreground",
  },
};

export const banner = (
  icon: string,
  content: string,
  variant: BannerVariant = "action",
  className?: string,
) => {
  const styles = bannerStyles[variant];
  return `
  <div class="${styles.bg} ring-1 ${styles.ring} rounded-2xl p-2 px-4 mt-2 flex items-start gap-2 text-body leading-5 font-medium ${styles.text} ${className || ""}">  
    <span class="flex-shrink-0">${icon}</span>
    <span>${content}</span>
  </div>
`;
};
