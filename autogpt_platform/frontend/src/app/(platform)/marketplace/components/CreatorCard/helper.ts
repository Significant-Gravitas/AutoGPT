const BACKGROUND_COLORS = [
  "bg-warning-surface border-warning/15",
  "bg-accent-surface border-brand/5",
  "bg-success-surface border-success/20",
  "bg-info-surface border-info/15",
];

export const backgroundColor = (index: number) =>
  BACKGROUND_COLORS[index % BACKGROUND_COLORS.length];
