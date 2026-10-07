// Stand-in for geist/font/sans and geist/font/mono. The geist package calls
// next/font/local from node_modules, which Storybook's Vite plugin does not
// transform, so stories load the same font files through @font-face.
import "./geist.css";

function font(family: string, variable: string, fallback: string) {
  return {
    className: "",
    variable: "",
    style: { fontFamily: `"${family}", ${fallback}` },
    cssVariable: variable,
  };
}

export const GeistSans = font(
  "Geist",
  "--font-geist-sans",
  "ui-sans-serif, system-ui, sans-serif",
);

export const GeistMono = font(
  "Geist Mono",
  "--font-geist-mono",
  "ui-monospace, SFMono-Regular, Menlo, monospace",
);
