import type { Icon as PhosphorIcon } from "@phosphor-icons/react";
import {
  ArrowRight,
  ArrowClockwise,
  Bell,
  Calendar,
  CaretDown,
  CaretRight,
  CheckCircle,
  CircleNotch,
  Clock,
  Copy,
  DownloadSimple,
  Folder,
  Gear,
  Heart,
  House,
  MagnifyingGlass,
  PlusCircle,
  Star,
  UploadSimple,
  User,
} from "@phosphor-icons/react/dist/ssr";

interface IconEntry {
  // Phosphor component used as the baseline fallback.
  phosphor: PhosphorIcon;
  // Matching export name in `@autogpt/icons` (stroke style), used when the
  // optional package is installed.
  autogpt: string;
}

// Maps a stable, semantic icon name to its Phosphor fallback and the equivalent
// `@autogpt/icons` export. Add new icons here — never import icon libraries
// directly in feature code, so the AutoGPT/Phosphor swap stays centralized.
export const iconRegistry = {
  home: { phosphor: House, autogpt: "HomeDefaultStroke" },
  search: { phosphor: MagnifyingGlass, autogpt: "SearchDefaultStroke" },
  settings: { phosphor: Gear, autogpt: "Settings01Stroke" },
  user: { phosphor: User, autogpt: "UserDefaultStroke" },
  add: { phosphor: PlusCircle, autogpt: "PlusCircleStroke" },
  check: { phosphor: CheckCircle, autogpt: "CheckTickCircleStroke" },
  calendar: { phosphor: Calendar, autogpt: "CalendarDefaultStroke" },
  clock: { phosphor: Clock, autogpt: "ClockDefaultStroke" },
  copy: { phosphor: Copy, autogpt: "CopyDefaultStroke" },
  folder: { phosphor: Folder, autogpt: "FolderDefaultStroke" },
  bell: { phosphor: Bell, autogpt: "NotificationBellOnStroke" },
  refresh: { phosphor: ArrowClockwise, autogpt: "RefreshStroke" },
  spinner: { phosphor: CircleNotch, autogpt: "SpinnerStroke" },
  star: { phosphor: Star, autogpt: "StarStroke" },
  heart: { phosphor: Heart, autogpt: "HeartStroke" },
  "chevron-down": { phosphor: CaretDown, autogpt: "ChevronDownStroke" },
  "chevron-right": { phosphor: CaretRight, autogpt: "ChevronRightStroke" },
  "arrow-right": { phosphor: ArrowRight, autogpt: "ArrowRightStroke" },
  download: { phosphor: DownloadSimple, autogpt: "DownloadDownStroke" },
  upload: { phosphor: UploadSimple, autogpt: "UploadUpStroke" },
} satisfies Record<string, IconEntry>;

export type IconName = keyof typeof iconRegistry;
