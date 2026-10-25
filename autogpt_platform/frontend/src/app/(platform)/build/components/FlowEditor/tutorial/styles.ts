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
    // TODO(tokens): no exact role: ring-violet-200 #DDD6FE
    ring: "ring-violet-200",
    // TODO(tokens): no exact role: text-violet-800 #5B21B6
    text: "text-violet-800",
  },
  info: {
    bg: "bg-info-surface",
    // TODO(tokens): no exact role: ring-blue-200 #BFDBFE
    ring: "ring-blue-200",
    // TODO(tokens): no exact role: text-blue-800 #1E40AF
    text: "text-blue-800",
  },
  warning: {
    bg: "bg-warning-surface",
    // TODO(tokens): no fitting role for ring-amber-200 #FDE68A; #FDE68A is warning-border (a border colour), and this is a ring (QUALITY 41(b))
    ring: "ring-amber-200",
    // TODO(tokens): no exact role: text-amber-800 #92400E
    text: "text-amber-800",
  },
  success: {
    bg: "bg-success-surface",
    // TODO(tokens): no exact role: ring-green-200 #94D5AC
    ring: "ring-green-200",
    // TODO(tokens): no exact role: text-green-800 #0C5A29
    text: "text-green-800",
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
