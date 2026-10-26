const BACKGROUND_COLORS = [
  // TODO(tokens): no exact role: border-amber-100/70 #FEF3C7
  "bg-warning-surface border-amber-100/70",
  // TODO(tokens): no exact role: border-violet-100/70 #EDE9FE
  "bg-accent-surface border-violet-100/70",
  // TODO(tokens): no exact role: border-green-100/70 #B7E2C7
  "bg-success-surface border-green-100/70",
  // TODO(tokens): no exact role: border-blue-100/70 #DBEAFE
  "bg-info-surface border-blue-100/70",
];

export const backgroundColor = (index: number) =>
  BACKGROUND_COLORS[index % BACKGROUND_COLORS.length];
