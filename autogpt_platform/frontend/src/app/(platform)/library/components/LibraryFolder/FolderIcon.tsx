import { motion } from "framer-motion";
import { Text } from "@/components/atoms/Text/Text";

type FolderSize = "xs" | "sm" | "md" | "lg" | "xl";
export type FolderColorName =
  | "neutral"
  | "slate"
  | "zinc"
  | "stone"
  | "red"
  | "orange"
  | "amber"
  | "yellow"
  | "lime"
  | "green"
  | "emerald"
  | "teal"
  | "cyan"
  | "sky"
  | "blue"
  | "indigo"
  | "violet"
  | "purple"
  | "fuchsia"
  | "pink"
  | "rose";

export type FolderColor = FolderColorName | (string & {});

const hexToColorName: Record<string, FolderColorName> = {
  "#3B82F6": "blue",
  "#3b82f6": "blue",
  "#A855F7": "purple",
  "#a855f7": "purple",
  "#10B981": "emerald",
  "#10b981": "emerald",
  "#F97316": "orange",
  "#f97316": "orange",
  "#EC4899": "pink",
  "#ec4899": "pink",
};

export function resolveColor(color: FolderColor | undefined): FolderColorName {
  if (!color) return "blue";
  if (color in hexToColorName) return hexToColorName[color];
  if (color in colorMap) return color as FolderColorName;
  return "blue";
}

interface Props {
  className?: string;
  size?: FolderSize | number;
  color?: FolderColor;
  icon?: string;
  isOpen?: boolean;
}

const sizeMap: Record<FolderSize, number> = {
  xs: 0.4,
  sm: 0.75,
  md: 1,
  lg: 1.25,
  xl: 1.5,
};

const colorMap: Record<
  FolderColorName,
  {
    bg: string;
    border: string;
    borderLight: string;
    fill: string;
    stroke: string;
  }
> = {
  neutral: {
    bg: "bg-neutral-300",
    border: "border-neutral-300",
    borderLight: "border-neutral-200",
    fill: "fill-neutral-300",
    stroke: "stroke-neutral-400",
  },
  slate: {
    bg: "bg-slate-300",
    border: "border-slate-300",
    borderLight: "border-slate-200",
    fill: "fill-slate-300",
    stroke: "stroke-slate-400",
  },
  zinc: {
    bg: "bg-zinc-300",
    border: "border-zinc-300",
    borderLight: "border-zinc-200",
    fill: "fill-zinc-300",
    stroke: "stroke-zinc-400",
  },
  stone: {
    bg: "bg-stone-300",
    border: "border-stone-300",
    borderLight: "border-stone-200",
    fill: "fill-stone-300",
    stroke: "stroke-stone-400",
  },
  red: {
    bg: "bg-red-300",
    border: "border-red-300",
    borderLight: "border-red-200",
    fill: "fill-red-300",
    stroke: "stroke-red-400",
  },
  orange: {
    bg: "bg-orange-200",
    border: "border-orange-200",
    borderLight: "border-orange-200",
    fill: "fill-orange-200",
    stroke: "stroke-orange-400",
  },
  amber: {
    bg: "bg-amber-200",
    border: "border-amber-200",
    borderLight: "border-amber-200",
    fill: "fill-amber-200",
    stroke: "stroke-amber-400",
  },
  yellow: {
    bg: "bg-yellow-200",
    border: "border-yellow-200",
    borderLight: "border-yellow-200",
    fill: "fill-yellow-200",
    stroke: "stroke-yellow-400",
  },
  lime: {
    bg: "bg-lime-300",
    border: "border-lime-300",
    borderLight: "border-lime-200",
    fill: "fill-lime-300",
    stroke: "stroke-lime-400",
  },
  green: {
    bg: "bg-green-200",
    border: "border-green-200",
    borderLight: "border-green-200",
    fill: "fill-green-200",
    stroke: "stroke-green-400",
  },
  emerald: {
    bg: "bg-emerald-300",
    border: "border-emerald-300",
    borderLight: "border-emerald-200",
    fill: "fill-emerald-300",
    stroke: "stroke-emerald-400",
  },
  teal: {
    bg: "bg-teal-300",
    border: "border-teal-300",
    borderLight: "border-teal-200",
    fill: "fill-teal-300",
    stroke: "stroke-teal-400",
  },
  cyan: {
    bg: "bg-cyan-300",
    border: "border-cyan-300",
    borderLight: "border-cyan-200",
    fill: "fill-cyan-300",
    stroke: "stroke-cyan-400",
  },
  sky: {
    bg: "bg-sky-300",
    border: "border-sky-300",
    borderLight: "border-sky-200",
    fill: "fill-sky-300",
    stroke: "stroke-sky-400",
  },
  blue: {
    bg: "bg-blue-300",
    border: "border-blue-300",
    borderLight: "border-blue-200",
    fill: "fill-blue-300",
    stroke: "stroke-blue-400",
  },
  indigo: {
    bg: "bg-indigo-300",
    border: "border-indigo-300",
    borderLight: "border-indigo-200",
    fill: "fill-indigo-300",
    stroke: "stroke-indigo-400",
  },
  violet: {
    bg: "bg-violet-300",
    border: "border-violet-300",
    borderLight: "border-violet-200",
    fill: "fill-violet-300",
    stroke: "stroke-violet-400",
  },
  purple: {
    bg: "bg-purple-200",
    border: "border-purple-200",
    borderLight: "border-purple-200",
    fill: "fill-purple-200",
    stroke: "stroke-purple-400",
  },
  fuchsia: {
    bg: "bg-fuchsia-300",
    border: "border-fuchsia-300",
    borderLight: "border-fuchsia-200",
    fill: "fill-fuchsia-300",
    stroke: "stroke-fuchsia-400",
  },
  pink: {
    bg: "bg-pink-300",
    border: "border-pink-300",
    borderLight: "border-pink-200",
    fill: "fill-pink-300",
    stroke: "stroke-pink-400",
  },
  rose: {
    bg: "bg-rose-300",
    border: "border-rose-300",
    borderLight: "border-rose-200",
    fill: "fill-rose-300",
    stroke: "stroke-rose-400",
  },
};

// Card-level bg (50) and border (200) classes per folder color
export const folderCardStyles: Record<
  FolderColorName,
  { bg: string; border: string }
> = {
  neutral: { bg: "bg-neutral-50", border: "border-neutral-200" },
  slate: { bg: "bg-slate-50", border: "border-slate-200" },
  zinc: { bg: "bg-zinc-50", border: "border-zinc-200" },
  stone: { bg: "bg-stone-50", border: "border-stone-200" },
  red: { bg: "bg-red-50", border: "border-red-200" },
  orange: { bg: "bg-orange-50", border: "border-orange-200" },
  amber: { bg: "bg-amber-50", border: "border-amber-200" },
  yellow: { bg: "bg-yellow-50", border: "border-yellow-200" },
  lime: { bg: "bg-lime-50", border: "border-lime-200" },
  green: { bg: "bg-green-50", border: "border-green-200" },
  emerald: { bg: "bg-emerald-50", border: "border-emerald-200" },
  teal: { bg: "bg-teal-50", border: "border-teal-200" },
  cyan: { bg: "bg-cyan-50", border: "border-cyan-200" },
  sky: { bg: "bg-sky-50", border: "border-sky-200" },
  blue: { bg: "bg-blue-50", border: "border-blue-200" },
  indigo: { bg: "bg-indigo-50", border: "border-indigo-200" },
  violet: { bg: "bg-violet-50", border: "border-violet-200" },
  purple: { bg: "bg-purple-50", border: "border-purple-200" },
  fuchsia: { bg: "bg-fuchsia-50", border: "border-fuchsia-200" },
  pink: { bg: "bg-pink-50", border: "border-pink-200" },
  rose: { bg: "bg-rose-50", border: "border-rose-200" },
};

export function FolderIcon({
  className = "",
  size = "xs",
  color = "blue",
  icon,
  isOpen = false,
}: Props) {
  const scale = typeof size === "number" ? size : sizeMap[size];
  const resolvedColor = resolveColor(color);
  const colors = colorMap[resolvedColor];

  return (
    <div
      className={`group relative cursor-pointer ${className}`}
      style={{
        width: 320 * scale,
        height: 208 * scale,
      }}
    >
      <div
        className="h-52 w-80 origin-top-left"
        style={{ transform: `scale(${scale})`, perspective: "500px" }}
      >
        <div
          className={`folder-back relative mx-auto flex h-full w-[87.5%] justify-center overflow-visible rounded-3xl ${colors.bg} ${colors.border}`}
        >
          {[
            {
              initial: { rotate: -3, x: -38, y: 2 },
              open: { rotate: -8, x: -70, y: -75 },
              transition: {
                type: "spring" as const,
                bounce: 0.15,
                stiffness: 160,
                damping: 22,
              },
              className: "z-10",
            },
            {
              initial: { rotate: 0, x: 0, y: 0 },
              open: { rotate: 1, x: 2, y: -95 },
              transition: {
                type: "spring" as const,
                duration: 0.55,
                bounce: 0.12,
                stiffness: 190,
                damping: 24,
              },
              className: "z-20",
            },
            {
              initial: { rotate: 3.5, x: 42, y: 1 },
              open: { rotate: 9, x: 75, y: -80 },
              transition: {
                type: "spring" as const,
                duration: 0.58,
                bounce: 0.17,
                stiffness: 170,
                damping: 21,
              },
              className: "z-10",
            },
          ].map((page, i) => (
            <motion.div
              key={i}
              initial={page.initial}
              animate={isOpen ? page.open : page.initial}
              transition={page.transition}
              className={`absolute top-2 h-fit w-32 rounded-xl shadow-lg ${page.className}`}
            >
              <Page color={resolvedColor} />
            </motion.div>
          ))}
        </div>

        <motion.div
          animate={{
            rotateX: isOpen ? -15 : 0,
          }}
          transition={{ type: "spring", duration: 0.5, bounce: 0.25 }}
          className="absolute inset-x-0 -bottom-px z-30 mx-auto flex h-44 w-[87.5%] origin-bottom items-end justify-center overflow-visible"
          style={{ transformStyle: "preserve-3d" }}
        >
          <svg
            className="h-auto w-full"
            viewBox="0 0 173 109"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            preserveAspectRatio="none"
          >
            <path
              className={`${colors.fill} ${colors.stroke}`}
              d="M15.0423 0.500003C0.5 0.500009 0.5 14.2547 0.5 14.2547V92.5C0.5 101.337 7.66344 108.5 16.5 108.5H156.5C165.337 108.5 172.5 101.337 172.5 92.5V34.3302C172.5 25.4936 165.355 18.3302 156.519 18.3302H108.211C98.1341 18.3302 91.2921 5.57144 82.0156 1.63525C80.3338 0.921645 78.2634 0.500002 75.7187 0.500003H15.0423Z"
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center text-7xl">
            {icon}
          </div>
        </motion.div>
      </div>
    </div>
  );
}

interface PageProps {
  color: FolderColorName;
}

function Page({ color = "blue" }: PageProps) {
  const colors = colorMap[color];
  return (
    <div
      className={`h-full w-full rounded-xl border bg-white p-4 ${colors.borderLight}`}
    >
      <div className="flex flex-col gap-2">
        <Text variant="h5" className="text-black">
          agent.json
        </Text>
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="flex gap-2">
            <div className="h-1.5 flex-1 rounded-full bg-neutral-100" />
            <div className="h-1.5 flex-1 rounded-full bg-neutral-100" />
          </div>
        ))}
      </div>
    </div>
  );
}
