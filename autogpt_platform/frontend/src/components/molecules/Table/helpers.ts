export const formatColumnTitle = (key: string): string => {
  return key.charAt(0).toUpperCase() + key.slice(1);
};

export const formatPlaceholder = (key: string): string => {
  return `Enter ${key.toLowerCase()}`;
};
