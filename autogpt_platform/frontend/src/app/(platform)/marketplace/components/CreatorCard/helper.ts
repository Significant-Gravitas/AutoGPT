const BACKGROUND_COLORS = [
  "bg-amber-100 dark:bg-amber-800", // #fef3c7 / #92400e
  "bg-violet-100 dark:bg-violet-800", // #ede9fe / #5b21b6
  "bg-green-100 dark:bg-green-800", // #dcfce7 / #065f46
  "bg-blue-100 dark:bg-blue-800", // #dbeafe / #1e3a8a
];
export const backgroundColor = (index: number) =>
  BACKGROUND_COLORS[index % BACKGROUND_COLORS.length];
