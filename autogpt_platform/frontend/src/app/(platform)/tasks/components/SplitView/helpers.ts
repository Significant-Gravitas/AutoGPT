export const PANEL_WIDTH_KEY = "tasks.panelWidth";
export const PANEL_DEFAULT_WIDTH = 480;
export const PANEL_MIN_WIDTH = 360;
export const PANEL_MAX_RATIO = 0.6;
export const PANEL_KEY_STEP = 24;

export const PANEL_GLIDE = {
  duration: 0.4,
  ease: [0.22, 1, 0.36, 1],
} as const;

export function maxPanelWidth(containerWidth: number) {
  if (containerWidth <= 0) return Number.POSITIVE_INFINITY;
  return Math.max(
    PANEL_MIN_WIDTH,
    Math.floor(containerWidth * PANEL_MAX_RATIO),
  );
}

export function clampPanelWidth(width: number, max: number) {
  return Math.round(Math.min(max, Math.max(PANEL_MIN_WIDTH, width)));
}

export function readStoredPanelWidth() {
  try {
    const raw = window.localStorage.getItem(PANEL_WIDTH_KEY);
    const value = raw ? Number(raw) : NaN;
    return Number.isFinite(value) ? value : PANEL_DEFAULT_WIDTH;
  } catch {
    return PANEL_DEFAULT_WIDTH;
  }
}

export function storePanelWidth(width: number) {
  try {
    window.localStorage.setItem(PANEL_WIDTH_KEY, String(width));
  } catch {
    // Storage can be unavailable (private mode, quota); the width just
    // doesn't survive a reload.
  }
}
