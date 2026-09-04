export interface ColorOption {
  id: string;
  label: string;
  // Written out in full so Tailwind's scanner keeps the class.
  swatchClassName: string;
  // Gradient stop for a surface washed in the expert's color.
  washFromClassName: string;
  // Border + tint for answers rendered in the expert's color.
  bubbleClassName: string;
  // Readable text at label sizes; the -300 swatch is too light for copy.
  textClassName: string;
  // Selected-card treatment: a deeper border over the lightest tint.
  selectedCardClassName: string;
  // Hover and focus treatment, so interaction never falls back to black.
  interactiveCardClassName: string;
  // Fill and outline for a selected row inside a grouped list, whose grey
  // dividers would otherwise be the only edge it has.
  rowSelectedClassName: string;
  // Ramp fed to the dither shader: shades 400 → 300 → 100 → white.
  ditherColors: readonly [string, string, string, string];
}

export const COLOR_OPTIONS: ColorOption[] = [
  {
    id: "rose-300",
    label: "Rose",
    swatchClassName: "bg-rose-300",
    washFromClassName: "from-rose-100",
    bubbleClassName: "border-rose-300 bg-rose-50",
    textClassName: "text-rose-700",
    selectedCardClassName: "border-rose-400 bg-rose-50 ring-2 ring-rose-200",
    interactiveCardClassName:
      "hover:border-rose-300 focus-within:ring-rose-200",
    rowSelectedClassName: "bg-rose-50 ring-1 ring-inset ring-rose-300",
    ditherColors: ["#fb7185", "#fda4af", "#ffe4e6", "#ffffff"],
  },
  {
    id: "red-300",
    label: "Red",
    swatchClassName: "bg-red-300",
    washFromClassName: "from-red-100",
    bubbleClassName: "border-red-300 bg-red-50",
    textClassName: "text-red-700",
    selectedCardClassName: "border-red-400 bg-red-50 ring-2 ring-red-200",
    interactiveCardClassName: "hover:border-red-300 focus-within:ring-red-200",
    rowSelectedClassName: "bg-red-50 ring-1 ring-inset ring-red-300",
    ditherColors: ["#f87171", "#fca5a5", "#fee2e2", "#ffffff"],
  },
  {
    id: "orange-300",
    label: "Orange",
    swatchClassName: "bg-orange-300",
    washFromClassName: "from-orange-100",
    bubbleClassName: "border-orange-300 bg-orange-50",
    textClassName: "text-orange-700",
    selectedCardClassName:
      "border-orange-400 bg-orange-50 ring-2 ring-orange-200",
    interactiveCardClassName:
      "hover:border-orange-300 focus-within:ring-orange-200",
    rowSelectedClassName: "bg-orange-50 ring-1 ring-inset ring-orange-300",
    ditherColors: ["#fb923c", "#fdba74", "#ffedd5", "#ffffff"],
  },
  {
    id: "amber-300",
    label: "Amber",
    swatchClassName: "bg-amber-300",
    washFromClassName: "from-amber-100",
    bubbleClassName: "border-amber-300 bg-amber-50",
    textClassName: "text-amber-700",
    selectedCardClassName: "border-amber-400 bg-amber-50 ring-2 ring-amber-200",
    interactiveCardClassName:
      "hover:border-amber-300 focus-within:ring-amber-200",
    rowSelectedClassName: "bg-amber-50 ring-1 ring-inset ring-amber-300",
    ditherColors: ["#fbbf24", "#fcd34d", "#fef3c7", "#ffffff"],
  },
  {
    id: "yellow-300",
    label: "Yellow",
    swatchClassName: "bg-yellow-300",
    washFromClassName: "from-yellow-100",
    bubbleClassName: "border-yellow-300 bg-yellow-50",
    textClassName: "text-yellow-700",
    selectedCardClassName:
      "border-yellow-400 bg-yellow-50 ring-2 ring-yellow-200",
    interactiveCardClassName:
      "hover:border-yellow-300 focus-within:ring-yellow-200",
    rowSelectedClassName: "bg-yellow-50 ring-1 ring-inset ring-yellow-300",
    ditherColors: ["#facc15", "#fde047", "#fef9c3", "#ffffff"],
  },
  {
    id: "lime-300",
    label: "Lime",
    swatchClassName: "bg-lime-300",
    washFromClassName: "from-lime-100",
    bubbleClassName: "border-lime-300 bg-lime-50",
    textClassName: "text-lime-700",
    selectedCardClassName: "border-lime-400 bg-lime-50 ring-2 ring-lime-200",
    interactiveCardClassName:
      "hover:border-lime-300 focus-within:ring-lime-200",
    rowSelectedClassName: "bg-lime-50 ring-1 ring-inset ring-lime-300",
    ditherColors: ["#a3e635", "#bef264", "#ecfccb", "#ffffff"],
  },
  {
    id: "green-300",
    label: "Green",
    swatchClassName: "bg-green-300",
    washFromClassName: "from-green-100",
    bubbleClassName: "border-green-300 bg-green-50",
    textClassName: "text-green-700",
    selectedCardClassName: "border-green-400 bg-green-50 ring-2 ring-green-200",
    interactiveCardClassName:
      "hover:border-green-300 focus-within:ring-green-200",
    rowSelectedClassName: "bg-green-50 ring-1 ring-inset ring-green-300",
    ditherColors: ["#4ade80", "#86efac", "#dcfce7", "#ffffff"],
  },
  {
    id: "emerald-300",
    label: "Emerald",
    swatchClassName: "bg-emerald-300",
    washFromClassName: "from-emerald-100",
    bubbleClassName: "border-emerald-300 bg-emerald-50",
    textClassName: "text-emerald-700",
    selectedCardClassName:
      "border-emerald-400 bg-emerald-50 ring-2 ring-emerald-200",
    interactiveCardClassName:
      "hover:border-emerald-300 focus-within:ring-emerald-200",
    rowSelectedClassName: "bg-emerald-50 ring-1 ring-inset ring-emerald-300",
    ditherColors: ["#34d399", "#6ee7b7", "#d1fae5", "#ffffff"],
  },
  {
    id: "teal-300",
    label: "Teal",
    swatchClassName: "bg-teal-300",
    washFromClassName: "from-teal-100",
    bubbleClassName: "border-teal-300 bg-teal-50",
    textClassName: "text-teal-700",
    selectedCardClassName: "border-teal-400 bg-teal-50 ring-2 ring-teal-200",
    interactiveCardClassName:
      "hover:border-teal-300 focus-within:ring-teal-200",
    rowSelectedClassName: "bg-teal-50 ring-1 ring-inset ring-teal-300",
    ditherColors: ["#2dd4bf", "#5eead4", "#ccfbf1", "#ffffff"],
  },
  {
    id: "cyan-300",
    label: "Cyan",
    swatchClassName: "bg-cyan-300",
    washFromClassName: "from-cyan-100",
    bubbleClassName: "border-cyan-300 bg-cyan-50",
    textClassName: "text-cyan-700",
    selectedCardClassName: "border-cyan-400 bg-cyan-50 ring-2 ring-cyan-200",
    interactiveCardClassName:
      "hover:border-cyan-300 focus-within:ring-cyan-200",
    rowSelectedClassName: "bg-cyan-50 ring-1 ring-inset ring-cyan-300",
    ditherColors: ["#22d3ee", "#67e8f9", "#cffafe", "#ffffff"],
  },
  {
    id: "sky-300",
    label: "Sky",
    swatchClassName: "bg-sky-300",
    washFromClassName: "from-sky-100",
    bubbleClassName: "border-sky-300 bg-sky-50",
    textClassName: "text-sky-700",
    selectedCardClassName: "border-sky-400 bg-sky-50 ring-2 ring-sky-200",
    interactiveCardClassName: "hover:border-sky-300 focus-within:ring-sky-200",
    rowSelectedClassName: "bg-sky-50 ring-1 ring-inset ring-sky-300",
    ditherColors: ["#38bdf8", "#7dd3fc", "#e0f2fe", "#ffffff"],
  },
  {
    id: "blue-300",
    label: "Blue",
    swatchClassName: "bg-blue-300",
    washFromClassName: "from-blue-100",
    bubbleClassName: "border-blue-300 bg-blue-50",
    textClassName: "text-blue-700",
    selectedCardClassName: "border-blue-400 bg-blue-50 ring-2 ring-blue-200",
    interactiveCardClassName:
      "hover:border-blue-300 focus-within:ring-blue-200",
    rowSelectedClassName: "bg-blue-50 ring-1 ring-inset ring-blue-300",
    ditherColors: ["#60a5fa", "#93c5fd", "#dbeafe", "#ffffff"],
  },
  {
    id: "indigo-300",
    label: "Indigo",
    swatchClassName: "bg-indigo-300",
    washFromClassName: "from-indigo-100",
    bubbleClassName: "border-indigo-300 bg-indigo-50",
    textClassName: "text-indigo-700",
    selectedCardClassName:
      "border-indigo-400 bg-indigo-50 ring-2 ring-indigo-200",
    interactiveCardClassName:
      "hover:border-indigo-300 focus-within:ring-indigo-200",
    rowSelectedClassName: "bg-indigo-50 ring-1 ring-inset ring-indigo-300",
    ditherColors: ["#818cf8", "#a5b4fc", "#e0e7ff", "#ffffff"],
  },
  {
    id: "violet-300",
    label: "Violet",
    swatchClassName: "bg-violet-300",
    washFromClassName: "from-violet-100",
    bubbleClassName: "border-violet-300 bg-violet-50",
    textClassName: "text-violet-700",
    selectedCardClassName:
      "border-violet-400 bg-violet-50 ring-2 ring-violet-200",
    interactiveCardClassName:
      "hover:border-violet-300 focus-within:ring-violet-200",
    rowSelectedClassName: "bg-violet-50 ring-1 ring-inset ring-violet-300",
    ditherColors: ["#a78bfa", "#c4b5fd", "#ede9fe", "#ffffff"],
  },
  {
    id: "fuchsia-300",
    label: "Fuchsia",
    swatchClassName: "bg-fuchsia-300",
    washFromClassName: "from-fuchsia-100",
    bubbleClassName: "border-fuchsia-300 bg-fuchsia-50",
    textClassName: "text-fuchsia-700",
    selectedCardClassName:
      "border-fuchsia-400 bg-fuchsia-50 ring-2 ring-fuchsia-200",
    interactiveCardClassName:
      "hover:border-fuchsia-300 focus-within:ring-fuchsia-200",
    rowSelectedClassName: "bg-fuchsia-50 ring-1 ring-inset ring-fuchsia-300",
    ditherColors: ["#e879f9", "#f0abfc", "#fae8ff", "#ffffff"],
  },
];

export function findColorOption(id: string | null) {
  return COLOR_OPTIONS.find((option) => option.id === id) ?? null;
}

export function ditherColorsFor(id: string | null) {
  return findColorOption(id)?.ditherColors;
}

export function bubbleClassFor(id: string | null) {
  return findColorOption(id)?.bubbleClassName;
}

export function swatchClassFor(id: string | null) {
  return findColorOption(id)?.swatchClassName;
}

export function washFromClassFor(id: string | null) {
  return findColorOption(id)?.washFromClassName;
}

export function textClassFor(id: string | null) {
  return findColorOption(id)?.textClassName;
}

export function selectedCardClassFor(id: string | null) {
  return findColorOption(id)?.selectedCardClassName;
}

export function interactiveCardClassFor(id: string | null) {
  return findColorOption(id)?.interactiveCardClassName;
}

export function rowSelectedClassFor(id: string | null) {
  return findColorOption(id)?.rowSelectedClassName;
}
