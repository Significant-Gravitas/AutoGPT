const BACKGROUND_COLORS = [
  "bg-violet-200 dark:bg-violet-800", // #ddd6fe / #5b21b6
  "bg-blue-200 dark:bg-blue-800", // #bfdbfe / #1e3a8a
  "bg-green-200 dark:bg-green-800", // #bbf7d0 / #065f46
];

export const getBackgroundColor = (index: number) => {
  return BACKGROUND_COLORS[index % BACKGROUND_COLORS.length];
};
