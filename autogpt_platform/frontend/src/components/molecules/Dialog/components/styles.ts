// Common styles as Tailwind class strings
const commonStyles = {
  title: "font-poppins text-md md:text-lg leading-none",
  overlay:
    "fixed inset-0 z-50 bg-stone-500/20 dark:bg-black/50 backdrop-blur-md animate-fade-in",
  content:
    "bg-stone-100 dark:bg-stone-800 p-6 fixed rounded-xl flex flex-col z-50 w-full overflow-hidden",
};

// Modal specific styles
export const modalStyles = {
  ...commonStyles,
  content: `${commonStyles.content} p-6 border border-stone-200 dark:border-stone-700 overflow-y-auto min-w-[40vw] max-w-[60vw] max-h-[95vh] top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-fadein`,
  iconWrap:
    "absolute top-2 right-3 bg-transparent p-2 rounded-full transition-colors duration-300 ease-in-out outline-none border-none",
  icon: "w-4 h-4 text-stone-800 dark:text-stone-300",
};

// Drawer specific styles
export const drawerStyles = {
  ...commonStyles,
  content: `${commonStyles.content} max-h-[90vh] w-full bottom-0 rounded-br-none rounded-bl-none`,
};
