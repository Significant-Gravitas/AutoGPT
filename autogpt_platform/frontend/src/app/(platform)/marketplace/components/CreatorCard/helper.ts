const BACKGROUND_COLORS = [
  // TODO(tokens): no exact role: border-warning/15 #FEF3C7
  "bg-warning-surface border-warning/15",
  // TODO(tokens): no exact role: border-brand/5 #EDE9FE
  "bg-accent-surface border-brand/5",
  // TODO(tokens): no exact role: border-success/20 #B7E2C7
  "bg-success-surface border-success/20",
  // TODO(tokens): no exact role: border-info/15 #DBEAFE
  "bg-info-surface border-info/15",
];

export const backgroundColor = (index: number) =>
  BACKGROUND_COLORS[index % BACKGROUND_COLORS.length];
