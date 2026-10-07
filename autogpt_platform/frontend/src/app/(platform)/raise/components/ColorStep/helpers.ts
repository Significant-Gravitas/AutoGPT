export interface ColorOption {
  id: string;
  label: string;
  // Written out in full so Tailwind's scanner keeps the class. Hues outside
  // the palette (rose, amber, lime, emerald, indigo, violet, fuchsia) are
  // written as hex: they are user-pickable identity colours and must stay
  // distinct from the palette's red, yellow, green and purple.
  swatchClassName: string;
  coverClassName: string;
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
    swatchClassName: "bg-[#fda4af]",
    coverClassName: "bg-[#fecdd3]",
    washFromClassName: "from-[#ffe4e6]",
    bubbleClassName: "border-[#fda4af] bg-[#fff1f2]",
    textClassName: "text-[#be123c]",
    selectedCardClassName:
      "border-[#fb7185] bg-[#fff1f2] ring-2 ring-[#fecdd3]",
    interactiveCardClassName:
      "hover:border-[#fda4af] focus-within:ring-[#fecdd3]",
    rowSelectedClassName: "bg-[#fff1f2] ring-1 ring-inset ring-[#fda4af]",
    ditherColors: ["#fb7185", "#fda4af", "#ffe4e6", "#ffffff"],
  },
  {
    id: "red-300",
    label: "Red",
    swatchClassName: "bg-red-300",
    coverClassName: "bg-red-200",
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
    coverClassName: "bg-orange-200",
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
    swatchClassName: "bg-[#fcd34d]",
    coverClassName: "bg-[#fde68a]",
    washFromClassName: "from-[#fef3c7]",
    bubbleClassName: "border-[#fcd34d] bg-[#fffbeb]",
    textClassName: "text-[#b45309]",
    selectedCardClassName:
      "border-[#fbbf24] bg-[#fffbeb] ring-2 ring-[#fde68a]",
    interactiveCardClassName:
      "hover:border-[#fcd34d] focus-within:ring-[#fde68a]",
    rowSelectedClassName: "bg-[#fffbeb] ring-1 ring-inset ring-[#fcd34d]",
    ditherColors: ["#fbbf24", "#fcd34d", "#fef3c7", "#ffffff"],
  },
  {
    id: "yellow-300",
    label: "Yellow",
    swatchClassName: "bg-yellow-300",
    coverClassName: "bg-yellow-200",
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
    swatchClassName: "bg-[#bef264]",
    coverClassName: "bg-[#d9f99d]",
    washFromClassName: "from-[#ecfccb]",
    bubbleClassName: "border-[#bef264] bg-[#f7fee7]",
    textClassName: "text-[#4d7c0f]",
    selectedCardClassName:
      "border-[#a3e635] bg-[#f7fee7] ring-2 ring-[#d9f99d]",
    interactiveCardClassName:
      "hover:border-[#bef264] focus-within:ring-[#d9f99d]",
    rowSelectedClassName: "bg-[#f7fee7] ring-1 ring-inset ring-[#bef264]",
    ditherColors: ["#a3e635", "#bef264", "#ecfccb", "#ffffff"],
  },
  {
    id: "green-300",
    label: "Green",
    swatchClassName: "bg-green-300",
    coverClassName: "bg-green-200",
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
    swatchClassName: "bg-[#6ee7b7]",
    coverClassName: "bg-[#a7f3d0]",
    washFromClassName: "from-[#d1fae5]",
    bubbleClassName: "border-[#6ee7b7] bg-[#ecfdf5]",
    textClassName: "text-[#047857]",
    selectedCardClassName:
      "border-[#34d399] bg-[#ecfdf5] ring-2 ring-[#a7f3d0]",
    interactiveCardClassName:
      "hover:border-[#6ee7b7] focus-within:ring-[#a7f3d0]",
    rowSelectedClassName: "bg-[#ecfdf5] ring-1 ring-inset ring-[#6ee7b7]",
    ditherColors: ["#34d399", "#6ee7b7", "#d1fae5", "#ffffff"],
  },
  {
    id: "teal-300",
    label: "Teal",
    swatchClassName: "bg-teal-300",
    coverClassName: "bg-teal-200",
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
    coverClassName: "bg-cyan-200",
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
    coverClassName: "bg-sky-200",
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
    coverClassName: "bg-blue-200",
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
    swatchClassName: "bg-[#a5b4fc]",
    coverClassName: "bg-[#c7d2fe]",
    washFromClassName: "from-[#e0e7ff]",
    bubbleClassName: "border-[#a5b4fc] bg-[#eef2ff]",
    textClassName: "text-[#4338ca]",
    selectedCardClassName:
      "border-[#818cf8] bg-[#eef2ff] ring-2 ring-[#c7d2fe]",
    interactiveCardClassName:
      "hover:border-[#a5b4fc] focus-within:ring-[#c7d2fe]",
    rowSelectedClassName: "bg-[#eef2ff] ring-1 ring-inset ring-[#a5b4fc]",
    ditherColors: ["#818cf8", "#a5b4fc", "#e0e7ff", "#ffffff"],
  },
  {
    id: "violet-300",
    label: "Violet",
    swatchClassName: "bg-[#c4b5fd]",
    coverClassName: "bg-[#ddd6fe]",
    washFromClassName: "from-[#ede9fe]",
    bubbleClassName: "border-[#c4b5fd] bg-[#f5f3ff]",
    textClassName: "text-[#6d28d9]",
    selectedCardClassName:
      "border-[#a78bfa] bg-[#f5f3ff] ring-2 ring-[#ddd6fe]",
    interactiveCardClassName:
      "hover:border-[#c4b5fd] focus-within:ring-[#ddd6fe]",
    rowSelectedClassName: "bg-[#f5f3ff] ring-1 ring-inset ring-[#c4b5fd]",
    ditherColors: ["#a78bfa", "#c4b5fd", "#ede9fe", "#ffffff"],
  },
  {
    id: "fuchsia-300",
    label: "Fuchsia",
    swatchClassName: "bg-[#f0abfc]",
    coverClassName: "bg-[#f5d0fe]",
    washFromClassName: "from-[#fae8ff]",
    bubbleClassName: "border-[#f0abfc] bg-[#fdf4ff]",
    textClassName: "text-[#a21caf]",
    selectedCardClassName:
      "border-[#e879f9] bg-[#fdf4ff] ring-2 ring-[#f5d0fe]",
    interactiveCardClassName:
      "hover:border-[#f0abfc] focus-within:ring-[#f5d0fe]",
    rowSelectedClassName: "bg-[#fdf4ff] ring-1 ring-inset ring-[#f0abfc]",
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

export function coverClassFor(id: string | null) {
  return findColorOption(id)?.coverClassName;
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
