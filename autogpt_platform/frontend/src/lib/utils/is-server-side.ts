export const isServerSide = (): boolean => {
  return typeof window === "undefined";
};
