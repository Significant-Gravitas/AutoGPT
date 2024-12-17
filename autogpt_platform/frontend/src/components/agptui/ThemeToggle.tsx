"use client";

import * as React from "react";
import { useTheme } from "next-themes";
import { IconMoon, IconSun } from "@/components/ui/icons";
import { Button } from "./Button";

export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = React.useState(false);

  React.useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <div className="relative h-6 w-12 rounded-full bg-gray-200 transition-colors dark:bg-gray-600">
        <div className="absolute left-0.5 top-0.5 h-5 w-5 rounded-full bg-white" />
      </div>
    );
  }

  return (
    <div
      className="relative h-6 w-12 cursor-pointer rounded-full bg-gray-200 transition-colors dark:bg-gray-600"
      onClick={() => setTheme(theme === "light" ? "dark" : "light")}
      role="button"
      tabIndex={0}
    >
      <div
        className={`absolute left-0.5 top-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-white transition-transform duration-200 ${
          theme === "dark" ? "translate-x-6" : ""
        }`}
      >
        {theme === "light" ? (
          <IconSun className="h-4 w-4 text-yellow-500" />
        ) : (
          <IconMoon className="h-4 w-4 text-gray-700" />
        )}
      </div>
      <span className="sr-only">Toggle theme</span>
    </div>
  );
}
