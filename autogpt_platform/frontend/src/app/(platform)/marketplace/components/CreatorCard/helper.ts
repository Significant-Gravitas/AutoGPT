const BACKGROUND_COLORS = [
  "bg-amber-50 border-amber-100/70",
  "bg-violet-50 border-violet-100/70",
  "bg-green-50 border-green-100/70",
  "bg-blue-50 border-blue-100/70",
];

export const backgroundColor = (index: number) =>
  BACKGROUND_COLORS[index % BACKGROUND_COLORS.length];
