export interface ColorOption {
  id: string;
  label: string;
  // Written out in full so Tailwind's scanner keeps the class.
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
    ditherColors: ["#fb7185", "#fda4af", "#ffe4e6", "#ffffff"],
  },
  {
    id: "red-300",
    label: "Red",
    swatchClassName: "bg-[#f69999]",
    coverClassName: "bg-[#fac5c5]",
    washFromClassName: "from-[#fdecec]",
    bubbleClassName: "border-[#f69999] bg-error-surface",
    textClassName: "text-error-foreground",
    selectedCardClassName:
      "border-[#f26969] bg-error-surface ring-2 ring-[#fac5c5]",
    interactiveCardClassName:
      "hover:border-[#f69999] focus-within:ring-[#fac5c5]",
    ditherColors: ["#f87171", "#fca5a5", "#fee2e2", "#ffffff"],
  },
  {
    id: "orange-300",
    label: "Orange",
    swatchClassName: "bg-[#feaf54]",
    coverClassName: "bg-[#ffc88a]",
    washFromClassName: "from-[#ffdab0]",
    bubbleClassName: "border-[#feaf54] bg-[#fff3e6]",
    textClassName: "text-[#b46000]",
    selectedCardClassName:
      "border-[#fe9f33] bg-[#fff3e6] ring-2 ring-[#ffc88a]",
    interactiveCardClassName:
      "hover:border-[#feaf54] focus-within:ring-[#ffc88a]",
    ditherColors: ["#fb923c", "#fdba74", "#ffedd5", "#ffffff"],
  },
  {
    id: "amber-300",
    label: "Amber",
    swatchClassName: "bg-[#fcd34d]",
    coverClassName: "bg-[#fde68a]",
    washFromClassName: "from-[#fef3c7]",
    bubbleClassName: "border-[#fcd34d] bg-warning-surface",
    textClassName: "text-warning-foreground",
    selectedCardClassName:
      "border-[#fbbf24] bg-warning-surface ring-2 ring-warning-border",
    interactiveCardClassName:
      "hover:border-[#fcd34d] focus-within:ring-warning-border",
    ditherColors: ["#fbbf24", "#fcd34d", "#fef3c7", "#ffffff"],
  },
  {
    id: "yellow-300",
    label: "Yellow",
    swatchClassName: "bg-[#f8d554]",
    coverClassName: "bg-[#fae28a]",
    washFromClassName: "from-[#fcebb0]",
    bubbleClassName: "border-[#f8d554] bg-[#fef9e6]",
    textClassName: "text-[#ae8800]",
    selectedCardClassName:
      "border-[#f7cd33] bg-[#fef9e6] ring-2 ring-[#fae28a]",
    interactiveCardClassName:
      "hover:border-[#f8d554] focus-within:ring-[#fae28a]",
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
    ditherColors: ["#a3e635", "#bef264", "#ecfccb", "#ffffff"],
  },
  {
    id: "green-300",
    label: "Green",
    swatchClassName: "bg-[#63c186]",
    coverClassName: "bg-[#94d5ac]",
    washFromClassName: "from-[#b7e2c7]",
    bubbleClassName: "border-[#63c186] bg-success-surface",
    textClassName: "text-success-foreground",
    selectedCardClassName:
      "border-[#45b56e] bg-success-surface ring-2 ring-[#94d5ac]",
    interactiveCardClassName:
      "hover:border-[#63c186] focus-within:ring-[#94d5ac]",
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
    ditherColors: ["#34d399", "#6ee7b7", "#d1fae5", "#ffffff"],
  },
  {
    id: "teal-300",
    label: "Teal",
    swatchClassName: "bg-[#5eead4]",
    coverClassName: "bg-[#99f6e4]",
    washFromClassName: "from-[#ccfbf1]",
    bubbleClassName: "border-[#5eead4] bg-[#f0fdfa]",
    textClassName: "text-[#0f766e]",
    selectedCardClassName:
      "border-[#2dd4bf] bg-[#f0fdfa] ring-2 ring-[#99f6e4]",
    interactiveCardClassName:
      "hover:border-[#5eead4] focus-within:ring-[#99f6e4]",
    ditherColors: ["#2dd4bf", "#5eead4", "#ccfbf1", "#ffffff"],
  },
  {
    id: "cyan-300",
    label: "Cyan",
    swatchClassName: "bg-[#67e8f9]",
    coverClassName: "bg-[#a5f3fc]",
    washFromClassName: "from-[#cffafe]",
    bubbleClassName: "border-[#67e8f9] bg-[#ecfeff]",
    textClassName: "text-[#0e7490]",
    selectedCardClassName:
      "border-[#22d3ee] bg-[#ecfeff] ring-2 ring-[#a5f3fc]",
    interactiveCardClassName:
      "hover:border-[#67e8f9] focus-within:ring-[#a5f3fc]",
    ditherColors: ["#22d3ee", "#67e8f9", "#cffafe", "#ffffff"],
  },
  {
    id: "sky-300",
    label: "Sky",
    swatchClassName: "bg-[#7dd3fc]",
    coverClassName: "bg-[#bae6fd]",
    washFromClassName: "from-[#e0f2fe]",
    bubbleClassName: "border-[#7dd3fc] bg-[#f0f9ff]",
    textClassName: "text-[#0369a1]",
    selectedCardClassName:
      "border-[#38bdf8] bg-[#f0f9ff] ring-2 ring-[#bae6fd]",
    interactiveCardClassName:
      "hover:border-[#7dd3fc] focus-within:ring-[#bae6fd]",
    ditherColors: ["#38bdf8", "#7dd3fc", "#e0f2fe", "#ffffff"],
  },
  {
    id: "blue-300",
    label: "Blue",
    swatchClassName: "bg-[#93c5fd]",
    coverClassName: "bg-[#bfdbfe]",
    washFromClassName: "from-[#dbeafe]",
    bubbleClassName: "border-[#93c5fd] bg-info-surface",
    textClassName: "text-[#1d4ed8]",
    selectedCardClassName:
      "border-[#60a5fa] bg-info-surface ring-2 ring-[#bfdbfe]",
    interactiveCardClassName:
      "hover:border-[#93c5fd] focus-within:ring-[#bfdbfe]",
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
    ditherColors: ["#818cf8", "#a5b4fc", "#e0e7ff", "#ffffff"],
  },
  {
    id: "violet-300",
    label: "Violet",
    swatchClassName: "bg-[#c4b5fd]",
    coverClassName: "bg-[#ddd6fe]",
    washFromClassName: "from-[#ede9fe]",
    bubbleClassName: "border-[#c4b5fd] bg-accent-surface",
    textClassName: "text-accent-strong",
    selectedCardClassName:
      "border-[#a78bfa] bg-accent-surface ring-2 ring-[#ddd6fe]",
    interactiveCardClassName:
      "hover:border-[#c4b5fd] focus-within:ring-[#ddd6fe]",
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

export function textClassFor(id: string | null) {
  return findColorOption(id)?.textClassName;
}

export function selectedCardClassFor(id: string | null) {
  return findColorOption(id)?.selectedCardClassName;
}

export function interactiveCardClassFor(id: string | null) {
  return findColorOption(id)?.interactiveCardClassName;
}
