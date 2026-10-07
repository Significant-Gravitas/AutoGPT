const BACKGROUND_COLORS = [
  "bg-yellow-50 border-yellow-100/70",
  "bg-purple-50 border-purple-100/70",
  "bg-green-50 border-green-100/70",
  "bg-blue-50 border-blue-100/70",
];

export const backgroundColor = (index: number) =>
  BACKGROUND_COLORS[index % BACKGROUND_COLORS.length];
