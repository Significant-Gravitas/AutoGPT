"use client";

import { createPortal } from "react-dom";
import {
  Fragment,
  createContext,
  useContext,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type DragEvent as ReactDragEvent,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
  type RefObject,
} from "react";
import { HugeiconsIcon, type IconSvgElement } from "@hugeicons/react";
import {
  ArrowDown01Icon,
  ArrowLeft01Icon,
  ArrowRight01Icon,
  ArrowUpRight01Icon,
  CommandLineIcon,
  Globe02Icon,
  MoreHorizontalIcon,
  NoteEditIcon,
  ArrowUpDownIcon,
  Cancel01Icon,
  DashboardSquare02Icon,
  File02Icon,
  FilterIcon,
  Folder01Icon,
  FolderOpenIcon,
  GridViewIcon,
  LeftToRightListBulletIcon,
  PencilEdit02Icon,
  Search01Icon,
  SidebarLeftIcon,
  SidebarRightIcon,
  Tick02Icon,
  PlusSignIcon,
  UserIcon,
  Settings01Icon,
  HelpCircleIcon,
  Logout01Icon,
  PinIcon,
  PinOffIcon,
  ViewIcon,
  Share08Icon,
  Copy01Icon,
  Archive02Icon,
  SquareArrowUpRight02Icon,
  ListViewIcon,
  Link01Icon,
  UserAdd01Icon,
  Analytics01Icon,
  CreditCardIcon,
  Activity01Icon,
  UserMultiple02Icon,
  InformationCircleIcon,
  ArrowLeftRightIcon,
  TranslateIcon,
  PaintBoardIcon,
  Plug01Icon,
  Download04Icon,
  Clock01Icon,
  Book02Icon,
  ComputerIcon,
  Sun03Icon,
  Moon02Icon,
  BubbleChatIcon,
  ArrowUp01Icon,
  Calendar03Icon,
  CheckListIcon,
  ClipboardIcon,
  ColumnDeleteIcon,
  Copy02Icon,
  CursorTextIcon,
  Delete02Icon,
  Dollar01Icon,
  EraserIcon,
  File01Icon,
  FileCodeIcon,
  Image01Icon,
  Table01Icon,
  FolderAddIcon,
  GridTableIcon,
  HardDriveIcon,
  Heading01Icon,
  Heading02Icon,
  Heading03Icon,
  InsertColumnLeftIcon,
  InsertColumnRightIcon,
  InsertRowDownIcon,
  InsertRowUpIcon,
  LeftToRightListNumberIcon,
  PencilEdit01Icon,
  PercentIcon,
  QuoteDownIcon,
  Redo02Icon,
  RemoveFormattingIcon,
  RowDeleteIcon,
  Scissor01Icon,
  SortByDown01Icon,
  SortByUp01Icon,
  Sorting05Icon,
  SortingAZ01Icon,
  SortingZA01Icon,
  SourceCodeIcon,
  TextBoldIcon,
  TextFontIcon,
  TextIcon,
  TextItalicIcon,
  TextNumberSignIcon,
  TextSelectionIcon,
  TextStrikethroughIcon,
  TextUnderlineIcon,
  Undo02Icon,
} from "@hugeicons/core-free-icons";
import {
  FilterMenu,
  type FilterGroup,
  type FilterValue,
} from "@/app/(platform)/admin/evals/affect/ui/filter-menu-clean";
import { TextField } from "@/app/(platform)/admin/evals/affect/ui/text-field-clean";
// The library's own primitives – the shell keeps its names and props and its class names as markers (screens style and
// look for .ui-as-skel, .ui-as-empty, .ui-as-popover, .ui-as-menu …)
import { Spinner as SpinnerItem } from "@/app/(platform)/admin/evals/affect/ui/spinner-clean";
import {
  Skeleton as SkeletonItem,
  Loading as LoadingItem,
} from "@/app/(platform)/admin/evals/affect/ui/skeleton-clean";
import { EmptyState as EmptyStateItem } from "@/app/(platform)/admin/evals/affect/ui/empty-state-clean";
import { IconButton as IconButtonItem } from "@/app/(platform)/admin/evals/affect/ui/icon-button-clean";
import { Popover as PopoverItem } from "@/app/(platform)/admin/evals/affect/ui/popover-clean";
import {
  DropdownMenu,
  type DropdownMenuItem,
} from "@/app/(platform)/admin/evals/affect/ui/dropdown-menu-clean";
import {
  SetupGuide,
  type SetupGroup,
} from "@/app/(platform)/admin/evals/affect/ui/setup-guide-clean";
import {
  CommandMenu,
  defaultGroups,
  type CommandGroup,
} from "@/app/(platform)/admin/evals/affect/ui/command-menu-clean";

/*
 * App Shell – the frame every affectUI screen sits in, in a single file.
 * A sidebar on a gray page (workspace, menu, projects, what's new, onboarding progress, user) and the main
 * area as a rounded white panel. The sidebar folds to a rail of its icons (each names itself on hover).
 * Drag the sidebar's edge to make it wider or narrower (between a minimum and a maximum; double-click resets it).
 * A work area on the right (`workArea`, or plain `panel` content as its one tab) – resizable and collapsible the
 * same way, with tabs like Codex: the screen's own tabs, artifacts and New tabs (+) with the start options of
 * WorkspacePanel (address field, tools, suggested sources). Every tab has its ×; closing the last folds the work area
 * and it comes back with the screen's default tabs. Artifacts open in shared viewers – a document you write in like
 * Word, a sheet you work in like Excel (formulas, a formula bar, Sum · Average · Count), a folder you browse like
 * Finder (Quick Look with Space), code and images – with right-click menus everywhere; your edits live on per file
 * while the screen lives. useWorkArea keeps that state in your screen, so the page can open artifacts too.
 * Two layouts: "inset" (the main area as a rounded panel on the gray page) and "flat" (no panel – the page runs edge
 * to edge next to the sidebar, split by a hairline). Pick one with `layout`, or for a whole app at once with
 * data-shell="flat" on any ancestor, e.g. <html> – like dark mode.
 * Also exports the small parts the screens share: icon, tabs, search, toolbar tools, section heads,
 * empty state and file cards. Icons: Hugeicons free set (stroke, rounded).
 */

/** Hugeicons, stroke + rounded with a slightly heavier stroke */
export const Icon = ({
  icon,
  size = 16,
  stroke = 1.9,
  color = "currentColor",
}: {
  icon: IconSvgElement;
  size?: number;
  stroke?: number;
  color?: string;
}) => (
  <HugeiconsIcon
    icon={icon}
    size={size}
    strokeWidth={stroke}
    color={color}
    aria-hidden
  />
);

/** Languages in the account menu, each with its flag */
const languages = [
  { label: "English", flag: "gb" },
  { label: "Deutsch", flag: "de" },
  { label: "Français", flag: "fr" },
  { label: "Español", flag: "es" },
] as const;

/** Small flags as drawings – the same on every system, also where flag emoji don't show */
/** Small flags for the language menus – drawn, since flag emoji look different on every system */
export const Flag = ({ code }: { code: "gb" | "de" | "fr" | "es" }) => (
  <svg className="ui-as-flag" viewBox="0 0 20 14" aria-hidden="true">
    {code === "de" && (
      <>
        <rect width="20" height="4.67" fill="#1f1f1f" />
        <rect y="4.67" width="20" height="4.67" fill="#dd0000" />
        <rect y="9.33" width="20" height="4.67" fill="#ffce00" />
      </>
    )}
    {code === "fr" && (
      <>
        <rect width="6.67" height="14" fill="#002395" />
        <rect x="6.67" width="6.67" height="14" fill="#ffffff" />
        <rect x="13.33" width="6.67" height="14" fill="#ed2939" />
      </>
    )}
    {code === "es" && (
      <>
        <rect width="20" height="14" fill="#aa151b" />
        <rect y="3.5" width="20" height="7" fill="#f1bf00" />
      </>
    )}
    {code === "gb" && (
      <>
        <rect width="20" height="14" fill="#012169" />
        <path d="M0 0 20 14M20 0 0 14" stroke="#ffffff" strokeWidth="2.8" />
        <path d="M0 0 20 14M20 0 0 14" stroke="#c8102e" strokeWidth="1" />
        <path d="M10 0v14M0 7h20" stroke="#ffffff" strokeWidth="4.2" />
        <path d="M10 0v14M0 7h20" stroke="#c8102e" strokeWidth="2.4" />
      </>
    )}
  </svg>
);

export type ShellNavItem = {
  label: string;
  icon: IconSvgElement;
  /** Small number at the end, e.g. unread items */ count?: number;
};
/** A labelled list under the menu – replaces Projects when given */
export type ShellGroup = {
  label: string;
  items: ShellNavItem[];
  /** Text shown while the list is empty */ empty?: string;
};

export const defaultNav: ShellNavItem[] = [
  { label: "New chat", icon: PencilEdit02Icon },
  { label: "Folders", icon: Folder01Icon },
  { label: "Integrations", icon: DashboardSquare02Icon },
];

export const defaultProjects = [
  "Q3 Campaign",
  "Brand assets",
  "Hiring",
  "Research",
];

/** Onboarding behind "Getting started" – shown in the library's Setup Guide */
export const defaultSetup: SetupGroup[] = [
  {
    id: "workspace",
    title: "Set up your workspace",
    tasks: [
      { id: "name", label: "Name your workspace", done: true },
      { id: "invite", label: "Invite your teammates" },
      { id: "connect", label: "Connect your first app" },
    ],
  },
  {
    id: "work",
    title: "Start working",
    tasks: [
      { id: "folder", label: "Create a folder" },
      { id: "chat", label: "Start your first chat" },
      { id: "upload", label: "Upload a file" },
    ],
  },
  {
    id: "profile",
    title: "Make it yours",
    tasks: [
      { id: "avatar", label: "Add a profile picture" },
      { id: "notify", label: "Choose your notifications" },
    ],
  },
];

/** Up to this width (of the app itself, not the window) the sidebar becomes a drawer */
const NARROW = 900;
/** Up to this app width the work area keeps the sizes it was given; on wider screens they grow in proportion */
const PANE_BASE = 1440;

const clamp = (v: number, min: number, max: number) =>
  Math.min(max, Math.max(min, v));

/** The first character as people see it – a whole emoji or accented letter, never half of one */
const firstChar = (text: string) => {
  const t = text.trim();
  if (!t) return "";
  const Segmenter = (
    Intl as unknown as {
      Segmenter?: new () => {
        segment: (s: string) => Iterable<{ segment: string }>;
      };
    }
  ).Segmenter;
  if (Segmenter)
    for (const part of new Segmenter().segment(t)) return part.segment;
  return Array.from(t)[0];
};
/** Counts in the sidebar stay short: 999+ */
const shortCount = (n: number) =>
  !Number.isFinite(n)
    ? ""
    : n > 999
      ? "999+"
      : String(Math.max(0, Math.round(n)));

const cheer = (pct: number) =>
  pct >= 100
    ? "All done!"
    : pct >= 60
      ? "Almost there!"
      : pct > 0
        ? "Great start!"
        : "Let's go!";

/** A chat in the sidebar – `folder` puts it inside one of the projects, which then open and close like folders */
export type ShellChat = {
  id: string;
  title: string;
  pinned?: boolean;
  unread?: boolean;
  folder?: string;
};

export type AppShellProps = {
  children: ReactNode;
  /** Label of the menu item or project that is open */
  active?: string;
  /** Called with the label of a menu item or project when it is clicked */
  onNavigate?: (label: string) => void;
  workspace?: string;
  /** Line under the workspace's name in its menu */
  workspaceMeta?: string;
  /** The account at the bottom – its menu shows the email under the name (made up from the name when left out) */
  user?: { name: string; avatar?: string; email?: string };
  nav?: ShellNavItem[];
  projects?: string[];
  /** Lists under the menu instead of Projects, e.g. Tools or Pipelines */
  groups?: ShellGroup[];
  /**
   * Chats in the sidebar: the projects turn into folders that open and close and hold their chats; the other chats
   * follow under Pinned and Chats. Pin, rename, mark as unread, move, copy or archive them from their ⋯ menu,
   * and drag them into another order or into a folder
   */
  chats?: ShellChat[];
  /** While the chats load: gray rows in their place (folders, chats, labels); the menu above them stays real */
  chatsLoading?: boolean;
  /** Called when a chat is clicked (default: onNavigate with its title) */
  onOpenChat?: (chat: ShellChat) => void;
  /** Tasks behind "Getting started" */
  setup?: SetupGroup[];
  /** Sidebar width in px – drag its edge between min and max; double-click the edge for the default again */
  sidebarWidth?: { default?: number; min?: number; max?: number };
  /**
   * The work area on the right of the main area, with tabs like Codex: the screen's own tabs, artifacts in the shared
   * viewers and New tabs with the start options – resizable and collapsible. Pass the options, or the result of
   * useWorkArea when the page opens artifacts itself
   */
  workArea?: WorkArea | WorkAreaOptions;
  /** Plain content for the work area – shown as its one tab, named by panelTitle (use `workspace` for more) */
  panel?: ReactNode;
  /**
   * A header across the whole app, like Claude's: the workspace and the sidebar toggle over the sidebar, the page's title and
   * actions over the page, and the work area's title over the work area – one bar, one line under it
   */
  header?: { title: ReactNode; actions?: ReactNode } | false;
  /** The cards at the bottom of the sidebar – what's new and Getting started (default: shown) */
  sideCards?: boolean;
  /** Heading of the right work area */
  panelTitle?: string;
  /** What the tab of a plain `panel` shows instead of panelTitle, e.g. "Brief / Summer launch" */
  panelHeading?: ReactNode;
  /**
   * Turns the sidebar into a settings sidebar: a back link with this title ("‹ Settings") in place of the workspace,
   * and no news or setup cards at the bottom. Clicking it calls onNavigate with `label` (default "Back to app").
   */
  back?: { title: string; label?: string };
  /** Start with the work area folded away – its icon waits in the top right corner (default: open) */
  defaultPanelOpen?: boolean;
  /** Width of the right work area in px, like sidebarWidth – on screens wider than 1440px the default and max grow in proportion */
  panelWidth?: { default?: number; min?: number; max?: number };
  /** "inset": the page as a rounded panel on the gray page; "flat": edge to edge, split by a hairline. Without it,
   *  data-shell="flat" on an ancestor makes it flat */
  layout?: "inset" | "flat";
  /**
   * A double sidebar, like Codex and ChatGPT: a column of icons on the gray page that always stays (the workspace on
   * top, these items, the account at the bottom), and the chats inside the main area – in a light gray column of their
   * own, split from the page by a line; it stays open and can be dragged wider or narrower. With it, the column's head
   * is a project switcher – "All projects" or one of `projects`, which narrows the chats to that project
   */
  rail?: {
    items: ShellNavItem[];
    /** Items at the bottom, above the account */ bottom?: ShellNavItem[];
    /** The rail item that is open */ active?: string;
  };
  className?: string;
};

/**
 * AutoGPT admin: the shell's look – its styles, variables and helpers (menus, popovers, toasts, empty states) – without
 * its sidebar or work area, for a screen hosted in a page that brings its own navigation.
 */
export function ShellFrame({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={["ui-as", className].filter(Boolean).join(" ")}>
      <style href="ui-app-shell" precedence="default">
        {css}
      </style>
      {children}
    </div>
  );
}

export function AppShell({
  children,
  active,
  onNavigate,
  workspace: workspaceName = "Acme",
  workspaceMeta = "Pro · 8 members",
  user = { name: "Alex Morgan" },
  nav = defaultNav,
  projects = defaultProjects,
  groups,
  chats,
  chatsLoading,
  onOpenChat,
  setup = defaultSetup,
  sidebarWidth,
  workArea,
  panel,
  panelTitle = "Workspace",
  panelHeading,
  back,
  defaultPanelOpen = true,
  panelWidth,
  header,
  sideCards = true,
  className,
  layout,
  rail: railNav,
}: AppShellProps) {
  const [news, setNews] = useState(2);
  const [collapsed, setCollapsed] = useState(false);
  const side = { default: 272, min: 220, max: 380, ...sidebarWidth };
  const base = { default: 360, min: 280, max: 560, ...panelWidth };
  const [sideW, setSideW] = useState(side.default);
  // The app's width: past PANE_BASE the work area's default and widest size grow with it, so it keeps its share
  const [appW, setAppW] = useState(0);
  const grow = Math.max(1, appW / PANE_BASE);
  const pane = {
    min: base.min,
    default: Math.round(base.default * grow),
    max: Math.round(base.max * grow),
  };
  // What you dragged it to – until then it follows the default
  const [paneDrag, setPaneDrag] = useState<number | null>(null);
  const paneW = Math.min(
    pane.max,
    Math.max(pane.min, paneDrag ?? pane.default),
  );
  const setPaneW = (w: number) => setPaneDrag(w);
  const [paneOpen, setPaneOpen] = useState(defaultPanelOpen);
  // The work area's tabs: the screen's workspace, or a plain panel as its one tab
  const given = isWorkArea(workArea) ? workArea : null;
  const own = useWorkArea(
    given
      ? undefined
      : ((workArea as WorkAreaOptions | undefined) ?? {
          tabs: [
            {
              id: "panel",
              title: panelTitle,
              label: panelHeading,
              content: panel,
            },
          ],
        }),
  );
  const ws = given ?? (workArea || panel ? own : null);
  const wsCount = ws?.state.tabs.length ?? 0;
  const wsReveal = ws?.state.reveal ?? 0;
  // While an edge is dragged the columns follow the pointer directly, without their glide
  const [resizing, setResizing] = useState(false);
  // Narrow (tablet, phone): the sidebar waits off screen and slides in as a drawer over the main area
  const root = useRef<HTMLDivElement>(null);
  const sideRef = useRef<HTMLElement>(null);
  const [narrow, setNarrow] = useState(false);
  const narrowNow = useRef(false);
  const [drawer, setDrawer] = useState(false);
  useLayoutEffect(() => {
    const el = root.current;
    if (!el) return;
    // Same width as the @container ui-as (max-width: 900px) rules below – measured in layout pixels, so a scaled preview stays wide
    const measure = () => {
      setAppW(el.offsetWidth);
      const next = el.offsetWidth <= NARROW;
      if (next === narrowNow.current) return;
      narrowNow.current = next;
      // All in one update, so the drawers never show (or take the focus) for a render in between
      setNarrow(next);
      if (!next) setDrawer(false);
      // Narrow: the right work area waits closed and slides in over the page when asked for
      setPaneOpen(!next && defaultPanelOpen);
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  // Collapsed on wide screens the sidebar stays as a rail of icons; on narrow ones it's a drawer that hides away
  const rail = collapsed && !narrow && !railNav;
  // The double sidebar's chats stay open – there is nothing to fold
  const sideHidden = narrow ? !drawer : false;
  // The project switcher in the double sidebar's head: all chats, or one project's
  const [project, setProject] = useState<string | null>(null);
  // Narrow: an open drawer is modal – the focus moves into it, the rest of the app waits (inert) behind the scrim,
  // and when it closes the focus goes back to the button that opened it
  const overlay = narrow
    ? drawer
      ? "side"
      : ws && paneOpen
        ? "panel"
        : null
    : null;
  const lastOverlay = useRef<typeof overlay>(null);
  useEffect(() => {
    const el = root.current;
    const was = lastOverlay.current;
    lastOverlay.current = overlay;
    if (!el || was === overlay) return;
    if (overlay) {
      const box = el.querySelector<HTMLElement>(
        overlay === "side" ? ":scope > .ui-as-side" : ":scope > .ui-as-panel",
      );
      if (box) firstFocusable(box)?.focus({ preventScroll: true });
    } else if (was) {
      const box = el.querySelector(
        was === "side" ? ":scope > .ui-as-side" : ":scope > .ui-as-panel",
      );
      const at = document.activeElement;
      if (!at || at === document.body || box?.contains(at))
        el.querySelector<HTMLElement>(
          was === "side"
            ? ":scope > .ui-as-side-open"
            : ":scope > .ui-as-panel-open",
        )?.focus({ preventScroll: true });
    }
  }, [overlay]);
  // Something asked to be seen in the work area – an artifact opened from the page: it opens if it's folded away
  useEffect(() => {
    if (wsReveal) setPaneOpen(true);
  }, [wsReveal]);
  // The last tab closed: the work area folds away and the focus moves to the button that brings it back – with the
  // screen's own tabs again
  const tabsBefore = useRef(wsCount);
  const foldFocus = useRef(false);
  useEffect(() => {
    const was = tabsBefore.current;
    tabsBefore.current = wsCount;
    if (!ws || was === 0 || wsCount > 0) return;
    foldFocus.current = true;
    setPaneOpen(false);
  }, [wsCount]);
  useEffect(() => {
    if (paneOpen || !foldFocus.current) return;
    foldFocus.current = false;
    root.current
      ?.querySelector<HTMLElement>(":scope > .ui-as-panel-open")
      ?.focus({ preventScroll: true });
  }, [paneOpen]);
  const openPane = () => {
    if (ws && ws.state.tabs.length === 0) ws.state.reset();
    setPaneOpen(true);
  };
  const go = (label: string) => {
    setDrawer(false);
    onNavigate?.(label);
  };

  // The search field opens the library's Command Menu over the app; ⌘K does too while the focus is inside the app
  const [search, setSearch] = useState(false);
  const searchBtn = useRef<HTMLButtonElement>(null);
  const closeSearch = () => {
    setSearch(false);
    searchBtn.current?.focus({ preventScroll: true });
  };
  // "Go to" lists the sidebar's own pages and projects; the other groups are the menu's samples
  const goTo: CommandGroup = {
    id: "go",
    label: "Go to",
    commands: [
      ...(groups ? groups.flatMap((g) => g.items) : nav),
      ...(groups
        ? []
        : projects.map((label) => ({ label, icon: FolderOpenIcon }))),
    ].map((item, i) => ({
      id: `go:${i}:${item.label}`,
      label: item.label,
      icon: <Icon icon={item.icon} size={16} />,
    })),
  };
  const commandGroups = defaultGroups.map((g) => (g.id === "go" ? goTo : g));
  // closed → open → closing → closed: while closing the guide stays, fades and the card shrinks, then the summary returns
  const [guide, setGuide] = useState<"closed" | "open" | "closing">("closed");
  const guideOpen = guide === "open";
  const setGuideOpen = (open: boolean) => {
    const at = document.activeElement;
    guideHadFocus.current =
      !!at && (at === document.body || !!guideInner.current?.contains(at));
    if (open) return setGuide("open");
    setGuide((g) => (g === "open" ? "closing" : g));
  };
  useEffect(() => {
    if (guide !== "closing") return;
    const t = window.setTimeout(() => setGuide("closed"), 420);
    return () => window.clearTimeout(t);
  }, [guide]);
  const guideInner = useRef<HTMLDivElement>(null);
  // The summary and the guide swap places – the focus moves with them, so the keyboard never drops to the page
  const guideHadFocus = useRef(false);
  useEffect(() => {
    const el = guideInner.current;
    if (!el) return;
    if (guide === "open" && guideHadFocus.current)
      firstFocusable(el)?.focus({ preventScroll: true });
    if (guide === "closed" && guideHadFocus.current)
      el.querySelector<HTMLElement>(".ui-as-steps")?.focus({
        preventScroll: true,
      });
    if (guide !== "closing") guideHadFocus.current = false;
  }, [guide]);
  const [guideHeight, setGuideHeight] = useState(0);
  const [closedHeight, setClosedHeight] = useState(0);
  // Read the content's height right when the state changes (not a frame later), so no step ever uses an old height;
  // the closed card's height is kept for the slot and for shrinking back
  useLayoutEffect(() => {
    const el = guideInner.current;
    if (!el) return;
    setGuideHeight(el.offsetHeight);
    if (guide === "closed") setClosedHeight(el.offsetHeight);
  }, [guide]);

  // The card's height follows its content, so opening, closing and folding groups all glide
  useLayoutEffect(() => {
    const el = guideInner.current;
    if (!el) return;
    const measure = () => setGuideHeight(el.offsetHeight);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!guideOpen) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && setGuideOpen(false);
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [guideOpen]);
  // Escape closes the drawer – unless the setup guide is open, then it closes the guide first
  useEffect(() => {
    if (!drawer || guide !== "closed") return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && setDrawer(false);
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [drawer, guide]);
  // Narrow: Escape closes the work area that slid in from the right
  useEffect(() => {
    if (!narrow || !paneOpen) return;
    const key = (e: KeyboardEvent) => e.key === "Escape" && setPaneOpen(false);
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [narrow, paneOpen]);
  const [done, setDone] = useState(() =>
    setup.flatMap((g) => g.tasks.filter((t) => t.done).map((t) => t.id)),
  );
  const total = setup.reduce((n, g) => n + g.tasks.length, 0);
  const pct = total ? Math.round((done.length / total) * 100) : 0;
  // First and last word, whole characters; empty names fall back to an icon and "Account"
  const workspace = workspaceName.trim() || "Workspace";
  const userName = user.name.trim();
  // Suffixes like "III" or "Jr." don't make an initial – "Christopher Montgomery III" is CM
  const words = userName
    .split(/\s+/)
    .filter(
      (w, i, all) =>
        w &&
        !(
          i > 0 &&
          i === all.length - 1 &&
          /^(jr\.?|sr\.?|i{2,3}|iv|v)$/i.test(w)
        ),
    );
  const initials = words.length
    ? (
        firstChar(words[0]) +
        (words.length > 1 ? firstChar(words[words.length - 1]) : "")
      ).toUpperCase()
    : "";
  // An avatar that doesn't load falls back to the initials
  const [avatarFailed, setAvatarFailed] = useState<string | null>(null);
  const showAvatar = user.avatar && avatarFailed !== user.avatar;
  // Who is signed in, for the head of the account menu – the email made up from the name when none is given
  const email =
    user.email ??
    `${(words[0] ?? "you").toLowerCase().replace(/[^a-z0-9]/g, "") || "you"}@${workspace.toLowerCase().replace(/[^a-z0-9]/g, "") || "example"}.com`;
  const [language, setLanguage] = useState("English");
  const [appearance, setAppearance] = useState("System");

  // The workspace switcher – in the sidebar's head, or with a header in the sidebar's part of it.
  // A settings sidebar shows a way back instead
  const backLink = back && (
    <button
      className="ui-as-side__back"
      type="button"
      aria-label={back.label ?? "Back to app"}
      onClick={() => go(back.label ?? "Back to app")}
    >
      <Icon icon={ArrowLeft01Icon} size={15} />
      <span>{back.title}</span>
    </button>
  );
  const workspaceItems: MenuItem[] = [
    {
      label: "Invite people",
      icon: UserAdd01Icon,
      onSelect: () => go("Members"),
    },
    {
      label: "Workspace settings",
      icon: Settings01Icon,
      divider: true,
      onSelect: () => go("Members"),
    },
    {
      label: "Analytics",
      icon: Analytics01Icon,
      onSelect: () => go("Analytics"),
    },
    { label: "Billing", icon: CreditCardIcon, onSelect: () => go("Billing") },
    {
      label: "Usage & limits",
      icon: Activity01Icon,
      onSelect: () => go("Billing"),
    },
    {
      label: "Members",
      icon: UserMultiple02Icon,
      onSelect: () => go("Members"),
    },
    {
      label: "Learn more",
      icon: InformationCircleIcon,
      divider: true,
      submenu: [
        { label: "Documentation", icon: Book02Icon },
        { label: "Changelog", icon: Clock01Icon },
        { label: "Help & feedback", icon: HelpCircleIcon },
      ],
    },
    {
      label: "Switch workspace",
      icon: ArrowLeftRightIcon,
      divider: true,
      submenu: [
        { label: workspace, description: workspaceMeta, checked: true },
        { label: "Personal", description: "Free", checked: false },
        { label: "Create workspace", icon: PlusSignIcon, divider: true },
      ],
    },
  ];
  const workspaceMenu = backLink || (
    <Dropdown
      className="ui-as-ws-menu"
      items={workspaceItems}
      trigger={({ open, toggle }) => (
        <button
          className="ui-as-side__workspace"
          type="button"
          aria-haspopup="menu"
          aria-expanded={open}
          onClick={toggle}
        >
          <span className="ui-as-side__logo">
            {firstChar(workspace).toUpperCase()}
          </span>
          <span className="ui-as-side__name" title={workspace}>
            {workspace}
          </span>
          <Icon icon={ArrowDown01Icon} size={13} />
        </button>
      )}
    />
  );

  // The account: in the sidebar's foot, or as a picture at the bottom of the double sidebar's icon column
  const accountMenu = (
    <Dropdown
      side="top"
      className={
        railNav ? "ui-as-user-menu ui-as-rail__account" : "ui-as-user-menu"
      }
      header={
        <span className="ui-as-account">
          {showAvatar ? (
            <img src={user.avatar} alt="" width={30} height={30} />
          ) : (
            <span className="ui-as-user__initials">
              {initials || <Icon icon={UserIcon} size={14} />}
            </span>
          )}
          <span className="ui-as-account__text">
            <b>{userName || "Account"}</b>
            <small>{email}</small>
          </span>
        </span>
      }
      items={[
        {
          label: "Language",
          icon: TranslateIcon,
          submenu: languages.map(({ label, flag }) => ({
            label,
            logo: <Flag code={flag} />,
            checked: label === language,
            onSelect: () => setLanguage(label),
          })),
        },
        {
          label: "Appearance",
          icon: PaintBoardIcon,
          submenu: (
            [
              ["System", ComputerIcon],
              ["Light", Sun03Icon],
              ["Dark", Moon02Icon],
            ] as const
          ).map(([m, icon]) => ({
            label: m,
            icon,
            checked: m === appearance,
            onSelect: () => setAppearance(m),
          })),
        },
        {
          label: "Integrations",
          icon: Plug01Icon,
          onSelect: () => go("Integrations"),
        },
        {
          label: "Settings",
          icon: Settings01Icon,
          onSelect: () => go("Profile"),
        },
        { label: "Download apps", icon: Download04Icon, divider: true },
        { label: "Changelog", icon: Clock01Icon, divider: true },
        {
          label: "Learn more",
          icon: InformationCircleIcon,
          submenu: [
            { label: "Documentation", icon: Book02Icon },
            { label: "Help & feedback", icon: HelpCircleIcon },
          ],
        },
        { label: "Log out", icon: Logout01Icon, divider: true },
      ]}
      trigger={({ open, toggle }) => (
        <button
          type="button"
          className="ui-as-user"
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label={railNav ? userName || "Account" : undefined}
          title={rail || railNav ? userName || "Account" : undefined}
          onClick={toggle}
        >
          {showAvatar ? (
            <img
              src={user.avatar}
              alt=""
              width={21}
              height={21}
              onError={() => setAvatarFailed(user.avatar ?? null)}
            />
          ) : (
            <span className="ui-as-user__initials">
              {initials || <Icon icon={UserIcon} size={12} />}
            </span>
          )}
          <span className="ui-as-user__name" title={userName || undefined}>
            {userName || "Account"}
          </span>
          <Icon icon={ArrowDown01Icon} size={13} />
        </button>
      )}
    />
  );

  // The double sidebar: its head is a project switcher, with search and folding as icons
  const projectItems: MenuItem[] = [
    {
      label: "All projects",
      icon: GridViewIcon,
      checked: !project,
      onSelect: () => setProject(null),
    },
    ...projects.map((name, i) => ({
      label: name,
      icon: FolderOpenIcon,
      checked: project === name,
      divider: i === 0,
      onSelect: () => setProject(name),
    })),
    {
      label: "New project",
      icon: FolderAddIcon,
      divider: true,
      onSelect: () => go("New project"),
    },
  ];
  const railHead = railNav && (
    <header className="ui-as-side__head">
      <Dropdown
        className="ui-as-project-menu"
        title="Projects"
        items={projectItems}
        trigger={({ open, toggle }) => (
          <button
            className="ui-as-side__workspace"
            type="button"
            aria-haspopup="menu"
            aria-expanded={open}
            onClick={toggle}
          >
            <span
              className="ui-as-side__name"
              title={project ?? "All projects"}
            >
              {project ?? "All projects"}
            </span>
            <Icon icon={ArrowDown01Icon} size={13} />
          </button>
        )}
      />
      <span className="ui-as-side__tools">
        <button
          ref={searchBtn}
          className="ui-as-icon-btn"
          type="button"
          aria-label="Search"
          aria-keyshortcuts="Meta+K"
          aria-haspopup="dialog"
          aria-expanded={search}
          title="Search ⌘K"
          onClick={() => setSearch(true)}
        >
          <Icon icon={Search01Icon} size={16} />
        </button>
        {narrow && (
          <button
            className="ui-as-icon-btn"
            type="button"
            aria-label="Close sidebar"
            onClick={() => setDrawer(false)}
          >
            <Icon icon={SidebarLeftIcon} size={16} />
          </button>
        )}
      </span>
    </header>
  );
  // The icon column: the workspace and the rail's items – the account at the bottom
  const railItem = (item: ShellNavItem, i: number) => (
    <button
      key={`${i}-${item.label}`}
      type="button"
      className="ui-as-rail__item"
      aria-label={item.label}
      title={item.label}
      aria-current={railNav?.active === item.label ? "page" : undefined}
      onClick={() => go(item.label)}
    >
      <Icon icon={item.icon} size={18} />
      {!!item.count && item.count > 0 && (
        <span className="ui-as-rail__dot" aria-hidden="true" />
      )}
    </button>
  );
  const railColumn = railNav && (
    <nav className="ui-as-rail" aria-label="Apps">
      <Dropdown
        className="ui-as-rail__ws"
        items={workspaceItems}
        trigger={({ open, toggle }) => (
          <button
            className="ui-as-rail__logo"
            type="button"
            aria-haspopup="menu"
            aria-expanded={open}
            aria-label={workspace}
            title={workspace}
            onClick={toggle}
          >
            {firstChar(workspace).toUpperCase()}
          </button>
        )}
      />
      <span className="ui-as-rail__line" aria-hidden="true" />
      {railNav.items.map(railItem)}
      <span className="ui-as-rail__spacer" />
      {railNav.bottom?.map(railItem)}
      {accountMenu}
    </nav>
  );
  const sideContent = (
    <>
      {railNav ? (
        railHead
      ) : (
        <>
          <header className="ui-as-side__head">
            {workspaceMenu}
            <span className="ui-as-side__tools">
              <button
                className="ui-as-icon-btn"
                type="button"
                aria-label={
                  narrow
                    ? "Close sidebar"
                    : rail
                      ? "Expand sidebar"
                      : "Collapse sidebar"
                }
                title={rail ? "Expand sidebar" : undefined}
                onClick={() =>
                  narrow ? setDrawer(false) : setCollapsed((c) => !c)
                }
              >
                <Icon icon={SidebarLeftIcon} size={16} />
              </button>
            </span>
          </header>

          {/* Search, like the affectUI library's: a field with its shortcut */}
          <button
            ref={searchBtn}
            className="ui-as-side__search"
            type="button"
            title={rail ? "Search ⌘K" : undefined}
            aria-keyshortcuts="Meta+K"
            aria-haspopup="dialog"
            aria-expanded={search}
            onClick={() => setSearch(true)}
          >
            <Icon icon={Search01Icon} size={15} />
            <span>Search…</span>
            <kbd>⌘K</kbd>
          </button>
        </>
      )}

      <nav className="ui-as-side__nav" aria-label="Main menu">
        <div className="ui-as-side__group">
          {nav.map((item, i) => (
            <button
              key={`${i}-${item.label}`}
              type="button"
              className="ui-as-side__item"
              aria-current={active === item.label ? "page" : undefined}
              title={rail ? item.label : undefined}
              onClick={() => go(item.label)}
            >
              <Icon icon={item.icon} size={16} />
              <span className="ui-as-side__text" title={item.label}>
                {item.label}
              </span>
              {!!item.count && item.count > 0 && (
                <span
                  className="ui-as-side__count"
                  title={item.count.toLocaleString()}
                >
                  {shortCount(item.count)}
                </span>
              )}
            </button>
          ))}
        </div>
        {groups ? (
          groups.map((group, gi) => (
            <div key={`${gi}-${group.label}`} className="ui-as-side__group">
              <span className="ui-as-side__label">{group.label}</span>
              {group.items.length === 0 && group.empty && (
                <span className="ui-as-side__empty">{group.empty}</span>
              )}
              {group.items.map((item, i) => (
                <button
                  key={`${i}-${item.label}`}
                  type="button"
                  className="ui-as-side__item"
                  aria-current={active === item.label ? "page" : undefined}
                  title={rail ? item.label : undefined}
                  onClick={() => go(item.label)}
                >
                  <Icon icon={item.icon} size={16} />
                  <span className="ui-as-side__text" title={item.label}>
                    {item.label}
                  </span>
                  {!!item.count && item.count > 0 && (
                    <span
                      className="ui-as-side__count"
                      title={item.count.toLocaleString()}
                    >
                      {shortCount(item.count)}
                    </span>
                  )}
                </button>
              ))}
            </div>
          ))
        ) : chats ? null : (
          <>
            <span className="ui-as-side__label">Projects</span>
            {projects.length === 0 && (
              <span className="ui-as-side__empty">No projects yet</span>
            )}
            {projects.map((name, i) => (
              <button
                key={`${i}-${name}`}
                type="button"
                className="ui-as-side__item"
                aria-current={active === name ? "page" : undefined}
                title={rail ? name : undefined}
                onClick={() => go(name)}
              >
                <Icon icon={FolderOpenIcon} size={16} />
                <span className="ui-as-side__text" title={name}>
                  {name}
                </span>
              </button>
            ))}
          </>
        )}
        {chats && chatsLoading ? (
          <ChatsSkeleton />
        ) : (
          chats && (
            <ChatList
              key={project ?? "all"}
              chats={
                project ? chats.filter((c) => c.folder === project) : chats
              }
              projects={project ? [project] : projects}
              loose={!project}
              rail={rail}
              active={active}
              side={sideRef}
              app={root}
              onOpen={(chat) =>
                onOpenChat ? onOpenChat(chat) : go(chat.title)
              }
              onOpenFolder={go}
            />
          )
        )}
      </nav>

      <div className="ui-as-side__foot">
        {!back && sideCards && news > 0 && (
          <div className="ui-as-news" data-count={news}>
            <div
              className="ui-as-news__card ui-as-news__card--back"
              aria-hidden="true"
            >
              <span className="ui-as-news__kicker">New in {workspace}</span>
              <Icon icon={Cancel01Icon} size={13} />
            </div>
            <div className="ui-as-news__card">
              <div className="ui-as-news__top">
                <span className="ui-as-news__kicker">New in {workspace}</span>
                <button
                  type="button"
                  className="ui-as-news__close"
                  aria-label="Dismiss"
                  onClick={() => {
                    // The last card goes away with its button – the focus moves on to Getting started (or the account)
                    if (news === 1)
                      refocus(
                        root.current?.querySelector<HTMLElement>(
                          ".ui-as-steps, .ui-as-user",
                        ),
                      );
                    setNews((n) => n - 1);
                  }}
                >
                  <Icon icon={Cancel01Icon} size={13} />
                </button>
              </div>
              <b className="ui-as-news__title">
                {news > 1 ? "Shared folders" : "Scheduled chats"}
              </b>
              <button type="button" className="ui-as-news__more">
                Full changelog
                <Icon icon={ArrowRight01Icon} size={13} />
              </button>
            </div>
          </div>
        )}

        {/* The card itself grows upward into the full setup guide – one block that morphs, no second panel */}
        {/* The slot keeps the closed card's height in the sidebar; the open card grows over the cards above it */}
        {!back && sideCards && total > 0 && (
          <div
            className="ui-as-guide-slot"
            style={closedHeight ? { height: closedHeight } : undefined}
          >
            <div
              className="ui-as-guide"
              data-open={guide !== "closed" || undefined}
              data-state={guide}
              style={{
                height:
                  (guide === "open" ? guideHeight : closedHeight) || undefined,
              }}
            >
              <div ref={guideInner} className="ui-as-guide__inner">
                {guide !== "closed" ? (
                  <div className="ui-as-guide__content" key="guide">
                    <SetupGuide
                      title="Getting started"
                      groups={setup.map((g) => ({
                        ...g,
                        tasks: g.tasks.map((t) => ({
                          ...t,
                          done: done.includes(t.id),
                        })),
                      }))}
                      onChange={setDone}
                      onClose={() => setGuideOpen(false)}
                    />
                  </div>
                ) : (
                  <button
                    type="button"
                    className="ui-as-steps ui-as-guide__content"
                    key="summary"
                    aria-expanded={false}
                    onClick={() => setGuideOpen(true)}
                  >
                    <span className="ui-as-steps__title">
                      Getting started
                      <Icon icon={ArrowRight01Icon} size={13} />
                    </span>
                    <span className="ui-as-steps__sub">
                      {pct}% done ·{" "}
                      <span className="ui-as-steps__cheer">{cheer(pct)}</span>
                    </span>
                    <span className="ui-as-steps__bar">
                      <i style={{ width: `${Math.max(pct, 4)}%` }} />
                    </span>
                  </button>
                )}
              </div>
            </div>
          </div>
        )}

        {!railNav && accountMenu}
      </div>
    </>
  );

  return (
    <WorkAreaContext.Provider value={ws}>
      <div
        ref={root}
        className={["ui-as", "ui-as-app", className].filter(Boolean).join(" ")}
        style={
          {
            "--side-w": `${sideW}px`,
            "--panel-w": `${paneW}px`,
          } as CSSProperties
        }
        data-collapsed={rail || undefined}
        data-rail={railNav ? "" : undefined}
        data-drawer={(narrow && drawer) || undefined}
        data-panel={ws ? (paneOpen ? "open" : "closed") : undefined}
        data-resizing={resizing || undefined}
        data-header={header ? "" : undefined}
        data-layout={layout}
        onKeyDown={(e) => {
          if (
            (e.metaKey || e.ctrlKey) &&
            e.key.toLowerCase() === "k" &&
            !search
          ) {
            e.preventDefault();
            e.stopPropagation();
            setSearch(true);
          }
        }}
      >
        <style href="ui-app-shell" precedence="default">
          {css}
        </style>
        {/* Search: the Command Menu over a soft scrim – a click outside or Esc in the empty field closes it */}
        {search && (
          <div
            className="ui-as-cmd"
            role="dialog"
            aria-modal="true"
            aria-label="Search"
            onPointerDown={(e) => e.target === e.currentTarget && closeSearch()}
          >
            <CommandMenu
              groups={commandGroups}
              autoFocus
              onClose={closeSearch}
              onRun={(id) => {
                closeSearch();
                if (id.startsWith("go:")) go(id.split(":").slice(2).join(":"));
              }}
            />
          </div>
        )}
        {/* With the sidebar hidden, the same icon sits in the top left corner of the main area to bring it back */}
        <button
          className="ui-as-icon-btn ui-as-side-open"
          type="button"
          aria-label="Open sidebar"
          tabIndex={sideHidden ? 0 : -1}
          inert={!!overlay}
          onClick={() => (narrow ? setDrawer(true) : setCollapsed(false))}
        >
          <Icon icon={SidebarLeftIcon} size={16} />
        </button>
        {/* The same for the work area on the right: its icon waits in the top right corner while it's closed */}
        {ws && (
          <button
            className="ui-as-icon-btn ui-as-panel-open"
            type="button"
            aria-label={`Open ${panelTitle.toLowerCase()}`}
            tabIndex={paneOpen ? -1 : 0}
            inert={!!overlay}
            onClick={openPane}
          >
            <Icon icon={SidebarRightIcon} size={16} />
          </button>
        )}
        {/* Narrow only: dims the main area behind the open drawer; a tap closes it */}
        <div
          className="ui-as-scrim"
          aria-hidden="true"
          onClick={() => {
            setDrawer(false);
            if (narrow) setPaneOpen(false);
          }}
        />
        {header && (
          <>
            <div className="ui-as-bar ui-as-bar--side" inert={sideHidden}>
              {workspaceMenu}
              <button
                className="ui-as-icon-btn"
                type="button"
                aria-label={rail ? "Expand sidebar" : "Collapse sidebar"}
                title={rail ? "Expand sidebar" : undefined}
                onClick={() => setCollapsed((c) => !c)}
              >
                <Icon icon={SidebarLeftIcon} size={16} />
              </button>
            </div>
            <div className="ui-as-bar ui-as-bar--main" inert={!!overlay}>
              <div className="ui-as-bar__title">{header.title}</div>
              {header.actions && (
                <div className="ui-as-bar__actions">{header.actions}</div>
              )}
            </div>
          </>
        )}
        <aside
          ref={sideRef}
          className="ui-as-side"
          inert={sideHidden || overlay === "panel"}
        >
          {railNav ? (
            <>
              {railColumn}
              <div className="ui-as-side__col">{sideContent}</div>
            </>
          ) : (
            sideContent
          )}
        </aside>
        {/* Drag handles on the edges of the main area – wide screens only; in the tab order after the sidebar and before the work area */}
        {!sideHidden && !narrow && !rail && (
          <Resizer
            edge="left"
            label="Resize sidebar"
            value={sideW}
            {...side}
            onChange={setSideW}
            onDrag={setResizing}
            root={root}
          />
        )}
        <main className="ui-as-main" inert={!!overlay}>
          {children}
        </main>
        {ws && paneOpen && !narrow && (
          <Resizer
            edge="right"
            label={`Resize ${panelTitle.toLowerCase()}`}
            value={paneW}
            {...pane}
            onChange={setPaneW}
            onDrag={setResizing}
            root={root}
          />
        )}
        {ws && (
          <aside
            className="ui-as-panel"
            aria-label={panelTitle}
            data-ws=""
            inert={!paneOpen || overlay === "side"}
          >
            <style href="ui-app-shell-artifacts" precedence="default">
              {artifactCss}
            </style>
            <header className="ui-as-panel__head">
              <WorkAreaTabs ws={ws} label={panelTitle} />
              <button
                className="ui-as-icon-btn"
                type="button"
                aria-label={`Close ${panelTitle.toLowerCase()}`}
                onClick={() => setPaneOpen(false)}
              >
                <Icon icon={SidebarRightIcon} size={16} />
              </button>
            </header>
            <div className="ui-as-panel__body">
              <WorkAreaBody ws={ws} />
            </div>
          </aside>
        )}
      </div>
    </WorkAreaContext.Provider>
  );
}

/**
 * The edge between the main area and a sidebar: drag it, use the arrow keys, or double-click it for the default width.
 * On hover the whole edge of the page darkens. The pointer's travel is divided by the app's scale, so a scaled-down
 * preview still follows the pointer exactly.
 */
function Resizer({
  edge,
  label,
  value,
  default: initial,
  min,
  max,
  onChange,
  onDrag,
  root,
}: {
  edge: "left" | "right";
  label: string;
  value: number;
  default: number;
  min: number;
  max: number;
  onChange: (width: number) => void;
  onDrag: (dragging: boolean) => void;
  root: RefObject<HTMLDivElement | null>;
}) {
  const start = useRef<{ x: number; width: number; scale: number } | null>(
    null,
  );
  const [dragging, setDragging] = useState(false);
  const sign = edge === "left" ? 1 : -1;
  const down = (e: ReactPointerEvent<HTMLDivElement>) => {
    if (e.button !== 0) return;
    const el = root.current;
    const scale =
      el && el.offsetWidth
        ? el.getBoundingClientRect().width / el.offsetWidth
        : 1;
    start.current = { x: e.clientX, width: value, scale: scale || 1 };
    e.currentTarget.setPointerCapture(e.pointerId);
    e.preventDefault();
    setDragging(true);
    onDrag(true);
  };
  const move = (e: ReactPointerEvent<HTMLDivElement>) => {
    const s = start.current;
    if (!s) return;
    onChange(
      clamp(
        Math.round(s.width + ((e.clientX - s.x) / s.scale) * sign),
        min,
        max,
      ),
    );
  };
  const up = () => {
    if (!start.current) return;
    start.current = null;
    setDragging(false);
    onDrag(false);
  };
  const key = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const step = e.shiftKey ? 48 : 16;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      onChange(
        clamp(value + (e.key === "ArrowRight" ? step : -step) * sign, min, max),
      );
    }
    if (e.key === "Home") onChange(min);
    if (e.key === "End") onChange(max);
  };
  return (
    <div
      className="ui-as-resizer"
      data-edge={edge}
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={value}
      aria-valuemin={min}
      aria-valuemax={max}
      tabIndex={0}
      data-dragging={dragging || undefined}
      onPointerDown={down}
      onPointerMove={move}
      onPointerUp={up}
      onPointerCancel={up}
      onLostPointerCapture={up}
      onDoubleClick={() => onChange(initial)}
      onKeyDown={key}
    />
  );
}

/* ---------- Shared parts ---------- */

/** Value that works controlled (value + onChange) or on its own (defaultValue) */
function useControlled<T>(
  value: T | undefined,
  fallback: T,
  onChange?: (v: T) => void,
): [T, (v: T) => void] {
  const [inner, setInner] = useState(fallback);
  return [value ?? inner, (v: T) => (setInner(v), onChange?.(v))];
}

/**
 * Tabs and radio groups are one stop for Tab: the arrow keys (and Home, End) move between their options and pick
 * the one they land on – focus follows, inside the group's own buttons
 */
export function rove<T extends string>(
  e: ReactKeyboardEvent<HTMLElement>,
  items: readonly T[],
  current: T,
  pick: (item: T) => void,
) {
  const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[
    e.key
  ];
  const at = items.indexOf(current);
  const next = step
    ? items[(at + step + items.length) % items.length]
    : e.key === "Home"
      ? items[0]
      : e.key === "End"
        ? items[items.length - 1]
        : undefined;
  if (next === undefined) return;
  e.preventDefault();
  pick(next);
  const buttons = [
    ...e.currentTarget.querySelectorAll<HTMLElement>(
      '[role="tab"], [role="radio"]',
    ),
  ];
  buttons[items.indexOf(next)]?.focus();
}

/** Segmented tabs: a soft gray pill slides to the active tab */
export function Tabs({
  items,
  value,
  defaultValue,
  onChange,
  controls,
}: {
  items: string[];
  value?: string;
  defaultValue?: string;
  onChange?: (value: string) => void;
  /** The id of the panel a tab shows, for aria-controls – leave out when there is none */
  controls?: (item: string) => string | undefined;
}) {
  const [current, setCurrent] = useControlled(
    value,
    defaultValue ?? items[0],
    onChange,
  );
  const list = useRef<HTMLDivElement>(null);
  const [pill, setPill] = useState<{ x: number; w: number } | null>(null);

  // The pill follows the active tab; measured from the layout, so it is right at any size
  useLayoutEffect(() => {
    const tab = list.current?.querySelector<HTMLElement>(
      '[aria-selected="true"]',
    );
    if (tab) setPill({ x: tab.offsetLeft, w: tab.offsetWidth });
  }, [current, items.join("|")]);

  return (
    <div
      ref={list}
      className="ui-as-tabs"
      role="tablist"
      onKeyDown={(e) => rove(e, items, current, setCurrent)}
    >
      {pill && (
        <span
          className="ui-as-tabs__pill"
          style={{ transform: `translateX(${pill.x}px)`, width: pill.w }}
          aria-hidden="true"
        />
      )}
      {items.map((item) => (
        <button
          key={item}
          type="button"
          role="tab"
          aria-selected={current === item}
          aria-controls={controls?.(item)}
          tabIndex={current === item || !items.includes(current) ? 0 : -1}
          className="ui-as-tabs__tab"
          onClick={() => setCurrent(item)}
        >
          {item}
        </button>
      ))}
    </div>
  );
}

/** The page's title in the header, with a chevron – a button, so it can open a menu (rename, move, delete…) */
export function BarTitle({
  children,
  onClick,
}: {
  children: ReactNode;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      className="ui-as-bar__pick"
      title={typeof children === "string" ? children : undefined}
      onClick={onClick}
    >
      <span>{children}</span>
      <Icon icon={ArrowDown01Icon} size={13} />
    </button>
  );
}

/** The library's Icon Button, secondary – white with the card's hairline shadow, like the toolbar's other tools */
export function IconButton({
  icon,
  label,
  onClick,
  pressed,
}: {
  icon: IconSvgElement;
  label: string;
  onClick?: () => void;
  pressed?: boolean;
}) {
  return (
    <IconButtonItem
      variant="secondary"
      icon={icon}
      label={label}
      pressed={pressed}
      onClick={onClick}
    />
  );
}

export type SortDir = "asc" | "desc";

/** Flips between A–Z and Z–A; the icon turns with it */
export function SortButton({
  value,
  onChange,
}: {
  value?: SortDir;
  onChange?: (dir: SortDir) => void;
}) {
  const [dir, setDir] = useControlled<SortDir>(value, "asc", onChange);
  return (
    <button
      type="button"
      className="ui-as-tool ui-as-sort"
      data-dir={dir}
      aria-label={
        dir === "asc"
          ? "Sorted A to Z – sort Z to A"
          : "Sorted Z to A – sort A to Z"
      }
      title={dir === "asc" ? "A–Z" : "Z–A"}
      onClick={() => setDir(dir === "asc" ? "desc" : "asc")}
    >
      <Icon icon={ArrowUpDownIcon} size={15} />
    </button>
  );
}

export type ViewMode = "grid" | "list";

export function ViewToggle({
  value,
  onChange,
}: {
  value?: ViewMode;
  onChange?: (view: ViewMode) => void;
}) {
  const [view, setView] = useControlled<ViewMode>(value, "grid", onChange);
  return (
    <div
      className="ui-as-view"
      role="radiogroup"
      aria-label="View"
      data-view={view}
      onKeyDown={(e) => rove(e, ["grid", "list"] as const, view, setView)}
    >
      <span className="ui-as-view__pill" aria-hidden="true" />
      {(
        [
          ["grid", GridViewIcon, "Grid"],
          ["list", LeftToRightListBulletIcon, "List"],
        ] as const
      ).map(([id, icon, label]) => (
        <button
          key={id}
          type="button"
          role="radio"
          aria-checked={view === id}
          tabIndex={view === id ? 0 : -1}
          aria-label={label}
          className="ui-as-view__btn"
          onClick={() => setView(id)}
        >
          <Icon icon={icon} size={15} />
        </button>
      ))}
    </div>
  );
}

export function SearchField({
  placeholder,
  width = 175,
  value,
  onChange,
}: {
  placeholder: string;
  width?: number;
  value?: string;
  onChange?: (value: string) => void;
}) {
  const [text, setText] = useControlled(value, "", onChange);
  return (
    // The library's Text Field: a search icon in front, × and Esc to clear
    <span className="ui-as-search" style={{ width }}>
      <TextField
        type="search"
        size="sm"
        prefix={<Icon icon={Search01Icon} size={15} />}
        placeholder={placeholder}
        aria-label={placeholder}
        value={text}
        onChange={(e) => setText(e.target.value)}
        clearable
        onClear={() => setText("")}
      />
    </span>
  );
}

/**
 * Small floating panel attached to its trigger – the library's Popover, unstyled: the class you pass draws it
 * (ui-as-menu for a menu card, the Filter Menu its own). It sits in the top layer, flips to stay on screen, closes on a
 * click outside, Escape or tabbing out and hands the focus back. `ui-as-popover` stays on it as a marker.
 */
export function Popover({
  open,
  onClose,
  children,
  align = "start",
  side = "bottom",
  className,
}: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  align?: "start" | "end";
  side?: "bottom" | "top" | "right";
  className?: string;
}) {
  return (
    <PopoverItem
      open={open}
      onClose={onClose}
      unstyled
      // To the right, its bottom edge lines up with the trigger's – as the shell's own did
      side={side}
      align={side === "right" ? "end" : align}
      offset={side === "right" ? 12 : 6}
      className={["ui-as-popover", className].filter(Boolean).join(" ")}
    >
      {children}
    </PopoverItem>
  );
}

export type MenuItem = {
  label: string;
  /** Second line in a lighter color */
  description?: string;
  icon?: IconSvgElement;
  /** A brand mark (Gmail, Slack …) in place of the icon, at 16px */
  logo?: ReactNode;
  /** Shows a check – for choices like a model or a period */
  checked?: boolean;
  /** Red text, e.g. Delete */
  danger?: boolean;
  /** A keyboard shortcut on the right, e.g. "⌥⌘R" */
  shortcut?: string;
  /** A line above this item, to group the ones after it */
  divider?: boolean;
  /** More choices that open to the side on hover or focus */
  submenu?: MenuItem[];
  onSelect?: () => void;
};

/** The items of one menu level – not the ones in its submenus */
const menuItems = (menu: Element | null) =>
  menu
    ? [
        ...menu.querySelectorAll<HTMLElement>(
          ':scope > [role^="menuitem"], :scope > .ui-as-menu__sub > [role^="menuitem"]',
        ),
      ]
    : [];

/** Focuses a menu's checked item, or its first one */
function focusMenu(menu: HTMLElement | null) {
  const items = menuItems(menu);
  (
    items.find((i) => i.getAttribute("aria-checked") === "true") ?? items[0]
  )?.focus({ preventScroll: true });
}

/**
 * Arrow keys move through a menu's items (wrapping), Home and End jump to the ends; on an item with a submenu
 * Arrow Right, Enter or Space step into it, Arrow Left steps back out
 */
function onMenuKey(e: ReactKeyboardEvent<HTMLDivElement>) {
  const menu = e.currentTarget;
  const items = menuItems(menu);
  const at = items.indexOf(document.activeElement as HTMLElement);
  const sub = (document.activeElement as HTMLElement | null)?.closest(
    ".ui-as-menu__sub",
  );
  const go = (i: number) => {
    e.preventDefault();
    e.stopPropagation();
    items[(i + items.length) % items.length]?.focus();
  };
  if (e.key === "ArrowDown") go(at + 1);
  else if (e.key === "ArrowUp") go(at < 0 ? -1 : at - 1);
  else if (e.key === "Home") go(0);
  else if (e.key === "End") go(-1);
  else if (
    at >= 0 &&
    sub?.parentElement === menu &&
    ["ArrowRight", "Enter", " "].includes(e.key)
  ) {
    e.preventDefault();
    e.stopPropagation();
    focusMenu(sub.querySelector<HTMLElement>('[role="menu"]'));
  } else if (
    e.key === "ArrowLeft" &&
    menu.parentElement?.classList.contains("ui-as-menu__subpanel")
  ) {
    e.preventDefault();
    e.stopPropagation();
    menu
      .closest(".ui-as-menu__sub")
      ?.querySelector<HTMLElement>(':scope > [role^="menuitem"]')
      ?.focus();
  }
}

/** List of actions or choices inside a popover */
export function MenuList({
  items,
  onPick,
}: {
  items: MenuItem[];
  onPick: () => void;
}) {
  return (
    <div role="menu" className="ui-as-menu__list" onKeyDown={onMenuKey}>
      {items.length === 0 && (
        <span className="ui-as-menu__empty">Nothing here yet</span>
      )}
      {items.map((item, i) => {
        const face = (
          <>
            {item.logo ? (
              <span className="ui-as-menu__logo" aria-hidden="true">
                {item.logo}
              </span>
            ) : (
              item.icon && <Icon icon={item.icon} size={15} />
            )}
            <span className="ui-as-menu__label" title={item.label}>
              <span className="ui-as-menu__text">{item.label}</span>
              {item.description && <small>{item.description}</small>}
            </span>
            {item.shortcut && (
              <kbd className="ui-as-menu__kbd">{item.shortcut}</kbd>
            )}
            {item.submenu && <Icon icon={ArrowRight01Icon} size={14} />}
            {item.checked && <Icon icon={Tick02Icon} size={14} />}
          </>
        );
        return (
          <Fragment key={`${i}-${item.label}`}>
            {item.divider && (
              <span className="ui-as-menu__sep" role="separator" />
            )}
            {item.submenu ? (
              // Opens to the side while the pointer or focus is on it
              <span className="ui-as-menu__sub">
                <button
                  type="button"
                  role="menuitem"
                  aria-haspopup="menu"
                  tabIndex={-1}
                  className="ui-as-menu__item"
                  // A click (or tap – no hover on touch) opens it too and keeps it open
                  onClick={(e) => {
                    const sub = e.currentTarget.parentElement;
                    sub?.parentElement
                      ?.querySelectorAll(":scope > .ui-as-menu__sub[data-open]")
                      .forEach(
                        (other) =>
                          other !== sub && other.removeAttribute("data-open"),
                      );
                    sub?.toggleAttribute("data-open");
                  }}
                >
                  {face}
                </button>
                <span className="ui-as-menu ui-as-menu__subpanel">
                  <MenuList items={item.submenu} onPick={onPick} />
                </span>
              </span>
            ) : (
              <button
                type="button"
                role={item.checked === undefined ? "menuitem" : "menuitemradio"}
                aria-checked={item.checked}
                tabIndex={-1}
                className="ui-as-menu__item"
                data-danger={item.danger || undefined}
                data-rich={item.description ? true : undefined}
                onClick={() => {
                  item.onSelect?.();
                  onPick();
                }}
              >
                {face}
              </button>
            )}
          </Fragment>
        );
      })}
    </div>
  );
}

/** The first control inside an element that is shown and reachable by Tab */
const firstFocusable = (root: Element) =>
  [
    ...root.querySelectorAll<HTMLElement>(
      "button, [href], input, select, textarea, [tabindex]",
    ),
  ].find(
    (el) =>
      el.tabIndex >= 0 &&
      !(el as HTMLButtonElement).disabled &&
      el.getClientRects().length > 0 &&
      getComputedStyle(el).visibility !== "hidden",
  );

/** After a pick the menu is gone – its trigger takes the focus back, unless the pick moved it somewhere (e.g. a rename field) */
const refocus = (trigger: HTMLElement | null | undefined) =>
  window.setTimeout(() => {
    const at = document.activeElement;
    if (trigger?.isConnected && (!at || at === document.body))
      trigger.focus({ preventScroll: true });
  });

/** Any trigger with a menu – the library's Dropdown Menu; the trigger gets the open state and a toggle */
export function Dropdown({
  trigger,
  items,
  align = "start",
  side = "bottom",
  title,
  header,
  className,
}: {
  trigger: (props: { open: boolean; toggle: () => void }) => ReactNode;
  items: MenuItem[];
  align?: "start" | "end";
  side?: "bottom" | "top";
  /** Small heading above the items */
  title?: string;
  /** Anything above the items, e.g. who is signed in */
  header?: ReactNode;
  className?: string;
}) {
  return (
    <DropdownMenu
      trigger={({ open, toggle }) => trigger({ open, toggle: () => toggle() })}
      items={items as unknown as DropdownMenuItem[]}
      align={align}
      side={side}
      title={title}
      header={header}
      className={["ui-as-anchor", className].filter(Boolean).join(" ")}
      menuClassName="ui-as-popover ui-as-menu"
    />
  );
}

/** Opens the library's Filter Menu in a popover; the button shows how many filters are on */
export function FilterButton({
  label,
  groups,
  value,
  onChange,
  align = "end",
}: {
  label: string;
  groups: FilterGroup[];
  value: FilterValue;
  onChange: (value: FilterValue) => void;
  align?: "start" | "end";
}) {
  const [open, setOpen] = useState(false);
  const count = Object.values(value).reduce((n, ids) => n + ids.length, 0);
  return (
    <span className="ui-as-anchor">
      <button
        type="button"
        className="ui-as-tool ui-as-tool--label"
        aria-expanded={open}
        data-active={count > 0 || undefined}
        onClick={() => setOpen((o) => !o)}
      >
        <Icon icon={FilterIcon} size={15} />
        {label}
        {count > 0 ? (
          <span className="ui-as-tool__count">{count}</span>
        ) : (
          <Icon icon={ArrowDown01Icon} size={13} />
        )}
      </button>
      <Popover
        open={open}
        onClose={() => setOpen(false)}
        align={align}
        className="ui-as-popover--filter"
      >
        <FilterMenu
          groups={groups}
          value={value}
          onChange={onChange}
          defaultOpen={groups.map((g) => g.id)}
          hints={false}
          title={label}
        />
      </Popover>
    </span>
  );
}

/** Plain menu of choices in a popover, e.g. a period picker */
export function MenuButton<T extends string>({
  value,
  options,
  onChange,
  icon,
  align = "end",
  className,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  icon?: IconSvgElement;
  align?: "start" | "end";
  className?: string;
}) {
  const current = options.find((o) => o.value === value);
  return (
    <DropdownMenu
      align={align}
      className="ui-as-anchor"
      menuClassName="ui-as-popover ui-as-menu"
      items={options.map((o) => ({
        label: o.label,
        checked: o.value === value,
        onSelect: () => onChange(o.value),
      }))}
      trigger={({ open, toggle, triggerProps }) => (
        <button
          {...triggerProps}
          type="button"
          className={["ui-as-tool ui-as-tool--label", className]
            .filter(Boolean)
            .join(" ")}
          aria-expanded={open}
          onClick={() => toggle()}
        >
          {icon && <Icon icon={icon} size={15} />}
          {current?.label}
          <Icon icon={ArrowDown01Icon} size={13} />
        </button>
      )}
    />
  );
}

/** Short confirmation at the bottom of the main area – "Saved to Reports" */
export function useToast() {
  const [toast, setToast] = useState<{ id: number; text: string } | null>(null);
  useEffect(() => {
    if (!toast) return;
    const t = window.setTimeout(() => setToast(null), 2200);
    return () => window.clearTimeout(t);
  }, [toast]);
  const show = (text: string) => setToast({ id: Date.now(), text });
  const node = toast ? (
    <div key={toast.id} className="ui-as-toast" role="status">
      <Icon icon={Tick02Icon} size={14} />
      <span className="ui-as-toast__text">{toast.text}</span>
    </div>
  ) : null;
  return [node, show] as const;
}

/** A section's head: a small uppercase title (with an optional count), then its tabs and tools */
export function SectionHead({
  title,
  count,
  left,
  right,
}: {
  title: string;
  count?: number;
  left?: ReactNode;
  right?: ReactNode;
}) {
  return (
    <div className="ui-as-section-head" data-inline={!left || undefined}>
      <h2 className="ui-as-section-head__title">
        {title}
        {count != null && (
          <span className="ui-as-section-head__count">{count}</span>
        )}
      </h2>
      {(left || right) && (
        <div className="ui-as-toolbar">
          <div className="ui-as-toolbar__left">{left}</div>
          <div className="ui-as-toolbar__right">{right}</div>
        </div>
      )}
    </div>
  );
}

/** The library's Empty State. `bare`: without the frame – the page's only content, e.g. inside .ui-as-center */
export function EmptyState({
  title,
  text,
  action,
  icon = File02Icon,
  bare,
}: {
  title: string;
  text: string;
  action?: ReactNode;
  icon?: IconSvgElement;
  tall?: boolean;
  bare?: boolean;
}) {
  return (
    <EmptyStateItem
      className="ui-as-empty"
      title={title}
      text={text}
      action={action}
      icon={icon}
      bare={bare}
    />
  );
}

/* ---------- Loading: one spinner, one skeleton for every screen ---------- */

/**
 * A small spinner – the library's Spinner: an arc on a faint ring in the text color, for something that works right
 * now (a run, a check, a connect). With `label` it is announced as a status; without it stays silent next to text
 * that says what happens. `className` lets a screen size or color it in place.
 */
export function Spinner({
  size = 14,
  label,
  className,
}: {
  size?: number;
  label?: string;
  className?: string;
}) {
  return (
    <SpinnerItem
      size={size}
      label={label}
      className={["ui-as-spin", className].filter(Boolean).join(" ")}
    />
  );
}

/**
 * A gray placeholder for text or a shape while the page's data loads – the library's Skeleton; a soft sheen sweeps
 * across it. Give it the size of what it stands in for, so nothing jumps when the data lands.
 */
export function Skeleton({
  width = "100%",
  height = 12,
  radius,
  className,
}: {
  width?: number | string;
  height?: number | string;
  radius?: number | string;
  className?: string;
}) {
  return (
    <SkeletonItem
      width={width}
      height={height}
      radius={radius}
      className={["ui-as-skel", className].filter(Boolean).join(" ")}
    />
  );
}

/**
 * Wraps the skeletons that stand in for one area – the library's Loading: role="status", aria-busy, fades in a beat
 * late. Pass the real layout's class as `className`, so the grid or list is the same one.
 */
export function Loading({
  label = "Loading",
  className,
  children,
}: {
  label?: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <LoadingItem
      label={label}
      className={["ui-as-loading", className].filter(Boolean).join(" ")}
    >
      {children}
    </LoadingItem>
  );
}

/* ---------- Chats in the sidebar ---------- */

/** The chats while they load: a label and gray rows for each section, in the rows' own size */
function ChatsSkeleton() {
  return (
    <Loading label="Loading chats" className="ui-as-chats-skel">
      {[
        [56, [148]],
        [52, [118, 132, 156, 112, 96, 108]],
        [36, [124, 104]],
      ].map(([label, rows], g) => (
        <div key={g} data-skel="">
          <span className="ui-as-side__label">
            <Skeleton width={label as number} height={8} />
          </span>
          {(rows as number[]).map((w, i) => (
            <span key={i} className="ui-as-side__item">
              <Skeleton width={w} height={10} />
            </span>
          ))}
        </div>
      ))}
    </Loading>
  );
}

/** Pinned chats on top, the rest below; each row has a ⋯ menu and a pin on hover, and can be dragged into place */
function ChatList({
  chats,
  projects,
  rail,
  active,
  side,
  app,
  onOpen,
  onOpenFolder,
  loose: showLoose = true,
}: {
  chats: ShellChat[];
  projects: string[];
  rail: boolean;
  /** The chats outside every project, under Chats – off while one project is picked */
  loose?: boolean;
  active?: string;
  side: RefObject<HTMLElement | null>;
  app: RefObject<HTMLDivElement | null>;
  onOpen: (chat: ShellChat) => void;
  /** Opens a project's own page */
  onOpenFolder: (name: string) => void;
}) {
  const [list, setList] = useState(chats);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [menu, setMenu] = useState<{
    id: string;
    left: number;
    below: number;
    above: number;
  } | null>(null);
  // What is being dragged: a chat (by id) or a folder (by name)
  const [drag, setDragState] = useState<{
    kind: "chat" | "folder";
    id: string;
  } | null>(null);
  const setDrag = (id: string | null) =>
    setDragState(id === null ? null : { kind: "chat", id });
  const [over, setOver] = useState<{ id: string; after: boolean } | null>(null);
  // The projects as folders, in their own order; pinned ones move up under Pinned
  const [folders, setFolders] = useState(() =>
    projects.map((name) => ({ name, pinned: false })),
  );
  // Folders that are open – the first one to start with
  const [openFolders, setOpenFolders] = useState(
    () => new Set(projects.slice(0, 1)),
  );
  const [dropFolder, setDropFolder] = useState<string | null>(null);
  const [renamingFolder, setRenamingFolder] = useState<string | null>(null);
  const toggleFolder = (name: string, open?: boolean) =>
    setOpenFolders((set) => {
      const next = new Set(set);
      if (open ?? !next.has(name)) next.add(name);
      else next.delete(name);
      return next;
    });
  const panel = useRef<HTMLDivElement>(null);
  const [toast, showToast] = useToast();

  const update = (id: string, change: Partial<ShellChat>) =>
    setList((l) => l.map((c) => (c.id === id ? { ...c, ...change } : c)));
  const togglePin = (chat: ShellChat) => {
    // A newly pinned chat goes to the end of the pinned ones, an unpinned one to the top of the rest
    setList((l) => {
      const rest = l.filter((c) => c.id !== chat.id);
      const moved = { ...chat, pinned: !chat.pinned };
      const at = moved.pinned
        ? rest.filter((c) => c.pinned).length
        : rest.findIndex((c) => !c.pinned);
      rest.splice(at < 0 ? rest.length : at, 0, moved);
      return rest;
    });
  };
  const archive = (chat: ShellChat) => {
    setList((l) => l.filter((c) => c.id !== chat.id));
    showToast(`“${chat.title}” archived`);
  };
  const copy = (text: string, done: string) => {
    navigator.clipboard?.writeText(text).catch(() => {});
    showToast(done);
  };

  // Drop a dragged chat before or after another one – into that one's section
  const drop = (targetId: string, after: boolean) => {
    if (drag?.kind !== "chat" || drag.id === targetId) return;
    const id = drag.id;
    setList((l) => {
      const moving = l.find((c) => c.id === id);
      const target = l.find((c) => c.id === targetId);
      if (!moving || !target) return l;
      const rest = l.filter((c) => c.id !== id);
      const at = rest.findIndex((c) => c.id === targetId) + (after ? 1 : 0);
      rest.splice(at, 0, {
        ...moving,
        pinned: target.pinned,
        folder: target.pinned ? moving.folder : target.folder,
      });
      return rest;
    });
  };
  // Dropped on a folder: the chat goes to the end of that folder, which opens
  const dropInto = (name: string) => {
    if (drag?.kind !== "chat") return;
    const id = drag.id;
    setList((l) => {
      const moving = l.find((c) => c.id === id);
      if (!moving) return l;
      return [
        ...l.filter((c) => c.id !== id),
        { ...moving, folder: name, pinned: false },
      ];
    });
    toggleFolder(name, true);
  };
  // A folder dropped before or after another one takes its place – and its section, pinned or not
  const moveFolder = (name: string, targetName: string, after: boolean) => {
    if (name === targetName) return;
    setFolders((f) => {
      const moving = f.find((x) => x.name === name);
      const target = f.find((x) => x.name === targetName);
      if (!moving || !target) return f;
      const rest = f.filter((x) => x.name !== name);
      rest.splice(
        rest.findIndex((x) => x.name === targetName) + (after ? 1 : 0),
        0,
        { ...moving, pinned: target.pinned },
      );
      return rest;
    });
  };
  const togglePinFolder = (name: string) =>
    setFolders((f) => {
      const moving = f.find((x) => x.name === name);
      if (!moving) return f;
      const rest = f.filter((x) => x.name !== name);
      const moved = { ...moving, pinned: !moving.pinned };
      const at = moved.pinned
        ? rest.filter((x) => x.pinned).length
        : rest.findIndex((x) => !x.pinned);
      rest.splice(at < 0 ? rest.length : at, 0, moved);
      return rest;
    });

  // A folder's own menu: rename it, move it between the sections, open its page, archive its chats or remove it
  const renameFolder = (from: string, to: string) => {
    const name = to.trim();
    setRenamingFolder(null);
    if (!name || name === from || folders.some((f) => f.name === name)) return;
    setFolders((f) => f.map((x) => (x.name === from ? { ...x, name } : x)));
    setList((l) =>
      l.map((c) => (c.folder === from ? { ...c, folder: name } : c)),
    );
    setOpenFolders(
      (set) => new Set([...set].map((x) => (x === from ? name : x))),
    );
  };
  const folderItems = (name: string, pinned: boolean): MenuItem[] => {
    const inside = list.filter((c) => c.folder === name);
    return [
      {
        label: pinned ? "Unpin" : "Pin",
        icon: pinned ? PinOffIcon : PinIcon,
        onSelect: () => togglePinFolder(name),
      },
      {
        label: "Rename",
        icon: PencilEdit02Icon,
        onSelect: () => setRenamingFolder(name),
      },
      {
        label: "Section",
        icon: ListViewIcon,
        divider: true,
        submenu: [
          {
            label: "Pinned",
            icon: PinIcon,
            checked: pinned,
            onSelect: () => !pinned && togglePinFolder(name),
          },
          {
            label: "Projects",
            icon: Folder01Icon,
            checked: !pinned,
            onSelect: () => pinned && togglePinFolder(name),
          },
        ],
      },
      {
        label: "Open project page",
        icon: FolderOpenIcon,
        onSelect: () => onOpenFolder(name),
      },
      {
        label: "Archive chats",
        icon: Archive02Icon,
        divider: true,
        onSelect: () => {
          setList((l) => l.filter((c) => c.folder !== name));
          showToast(
            inside.length
              ? `${inside.length} ${inside.length === 1 ? "chat" : "chats"} archived`
              : "No chats to archive",
          );
        },
      },
      {
        label: "Remove project",
        icon: Cancel01Icon,
        divider: true,
        onSelect: () => {
          // Its chats stay – they move to the loose chats
          setFolders((f) => f.filter((x) => x.name !== name));
          setList((l) =>
            l.map((c) => (c.folder === name ? { ...c, folder: undefined } : c)),
          );
          showToast(`“${name}” removed`);
        },
      },
    ];
  };

  // The picture that follows the pointer: a copy of the row on the card's white with rounded corners, no edge, drawn outside
  // the scrolling list, so nothing shows through it and nothing cuts it off
  const lift = (
    e: ReactDragEvent<HTMLElement>,
    label: string,
    start: () => void,
  ) => {
    const row = e.currentTarget;
    const item = row.querySelector<HTMLElement>(".ui-as-side__item") ?? row;
    const look = getComputedStyle(item);
    const tokens = getComputedStyle(row);
    const ghost = item.cloneNode(true) as HTMLElement;
    ghost.querySelector(".ui-as-chat-folder__arrow")?.remove();
    Object.assign(ghost.style, {
      position: "fixed",
      top: "-1000px",
      left: "0",
      width: `${item.offsetWidth}px`,
      height: `${item.offsetHeight}px`,
      display: "flex",
      alignItems: "center",
      gap: look.gap,
      padding: look.padding,
      boxSizing: "border-box",
      borderRadius: "8px",
      background: tokens.getPropertyValue("--card").trim() || "#ffffff",
      border: "0",
      outline: "0",
      boxShadow: "none",
      font: look.font,
      color: look.color,
      whiteSpace: "nowrap",
      overflow: "hidden",
      pointerEvents: "none",
    });
    ghost
      .querySelectorAll<HTMLElement>("svg")
      .forEach((svg) => (svg.style.flex = "none"));
    document.body.appendChild(ghost);
    const box = row.getBoundingClientRect();
    e.dataTransfer.setDragImage(
      ghost,
      Math.min(e.clientX - box.left, item.offsetWidth),
      e.clientY - box.top,
    );
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", label);
    setMenu(null);
    // The browser takes the picture right after this handler – then the copy goes and the row fades in its place
    window.setTimeout(() => {
      ghost.remove();
      start();
    });
  };
  const endDrag = () => {
    setDragState(null);
    setOver(null);
    setDropFolder(null);
  };

  // The menu opens under its ⋯ button – drawn in the sidebar, so the scrolling list doesn't cut it off – or above it near the bottom
  const openMenu = (id: string, button: HTMLElement) => {
    const box = side.current?.getBoundingClientRect();
    if (!box || !side.current) return;
    const r = button.getBoundingClientRect();
    const scale = box.width / side.current.offsetWidth || 1;
    setMenu((m) =>
      m?.id === id
        ? null
        : {
            id,
            left: (r.left - box.left) / scale,
            below: (r.bottom - box.top) / scale + 4,
            above: (r.top - box.top) / scale - 4,
          },
    );
  };
  useLayoutEffect(() => {
    const el = panel.current;
    if (!menu || !el || !side.current) return;
    const room = side.current.offsetHeight - 8;
    el.style.top = `${menu.below + el.offsetHeight > room ? Math.max(8, menu.above - el.offsetHeight) : menu.below}px`;
    // Never past the sidebar's right edge – on a phone the drawer is narrower than the menu's usual reach
    el.style.left = `${Math.max(8, Math.min(menu.left, side.current.offsetWidth - el.offsetWidth - 8))}px`;
  }, [menu, side]);
  // The menu's ⋯ button – the focus returns to it when the menu closes from the keyboard
  const menuButton = (id: string) =>
    side.current?.querySelector<HTMLElement>(
      `[data-chat-menu="${CSS.escape(id)}"]`,
    );
  const menuId = menu?.id;
  useEffect(() => {
    if (menuId)
      focusMenu(
        panel.current?.querySelector<HTMLElement>('[role="menu"]') ?? null,
      );
  }, [menuId]);
  useEffect(() => {
    if (!menu) return;
    const away = (e: PointerEvent) => {
      const t = e.target as Element;
      if (
        !panel.current?.contains(t) &&
        !t.closest?.(`[data-chat-menu="${CSS.escape(menu.id)}"]`)
      )
        setMenu(null);
    };
    // First in line, so the drawer around it stays open
    const key = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopPropagation();
      if (panel.current?.contains(document.activeElement))
        menuButton(menu.id)?.focus({ preventScroll: true });
      setMenu(null);
    };
    const scroll = () => setMenu(null);
    const nav = side.current?.querySelector(".ui-as-side__nav");
    document.addEventListener("pointerdown", away);
    window.addEventListener("keydown", key, true);
    nav?.addEventListener("scroll", scroll);
    return () => {
      document.removeEventListener("pointerdown", away);
      window.removeEventListener("keydown", key, true);
      nav?.removeEventListener("scroll", scroll);
    };
  }, [menu, side]);

  const itemsFor = (chat: ShellChat): MenuItem[] => [
    {
      label: "Rename",
      icon: PencilEdit02Icon,
      shortcut: "⌥⌘R",
      onSelect: () => setRenaming(chat.id),
    },
    {
      label: chat.pinned ? "Unpin" : "Pin",
      icon: chat.pinned ? PinOffIcon : PinIcon,
      shortcut: "⌥⌘P",
      onSelect: () => togglePin(chat),
    },
    {
      label: chat.unread ? "Mark as read" : "Mark as unread",
      icon: ViewIcon,
      shortcut: "⇧⌘U",
      onSelect: () => update(chat.id, { unread: !chat.unread }),
    },
    {
      label: "Project",
      icon: Folder01Icon,
      submenu: [
        ...folders.map(({ name }) => ({
          label: name,
          icon: FolderOpenIcon,
          checked: chat.folder === name,
          onSelect: () => {
            update(chat.id, { folder: name });
            if (!chat.pinned) toggleFolder(name, true);
          },
        })),
        {
          label: "No project",
          divider: true,
          checked: !chat.folder,
          onSelect: () => update(chat.id, { folder: undefined }),
        },
      ],
    },
    {
      label: "Section",
      icon: ListViewIcon,
      submenu: [
        {
          label: "Pinned",
          icon: PinIcon,
          checked: !!chat.pinned,
          onSelect: () => !chat.pinned && togglePin(chat),
        },
        {
          label: "Chats",
          icon: BubbleChatIcon,
          checked: !chat.pinned,
          onSelect: () => chat.pinned && togglePin(chat),
        },
      ],
    },
    {
      label: "Share",
      icon: Share08Icon,
      divider: true,
      onSelect: () =>
        copy(`${location.origin}/chat/${chat.id}`, "Share link copied"),
    },
    {
      label: "Copy",
      icon: Copy01Icon,
      submenu: [
        {
          label: "Copy link",
          icon: Link01Icon,
          onSelect: () =>
            copy(`${location.origin}/chat/${chat.id}`, "Link copied"),
        },
        {
          label: "Copy as Markdown",
          icon: File02Icon,
          onSelect: () => copy(`# ${chat.title}`, "Chat copied as Markdown"),
        },
        {
          label: "Copy chat ID",
          icon: Copy01Icon,
          onSelect: () => copy(chat.id, "Chat ID copied"),
        },
      ],
    },
    {
      label: "Open in new window",
      icon: SquareArrowUpRight02Icon,
      divider: true,
      onSelect: () => showToast("Opening in a new window"),
    },
    {
      label: "Archive",
      icon: Archive02Icon,
      shortcut: "⇧⌘A",
      divider: true,
      onSelect: () => archive(chat),
    },
  ];

  // The menu's shortcuts work on the focused row
  const onKey =
    (chat: ShellChat) => (e: ReactKeyboardEvent<HTMLDivElement>) => {
      if (!e.metaKey || renaming) return;
      const run = e.altKey
        ? { KeyR: () => setRenaming(chat.id), KeyP: () => togglePin(chat) }[
            e.code
          ]
        : e.shiftKey
          ? {
              KeyU: () => update(chat.id, { unread: !chat.unread }),
              KeyA: () => archive(chat),
            }[e.code]
          : undefined;
      if (run) {
        e.preventDefault();
        run();
      }
    };

  const folderNames = folders.map((f) => f.name);
  const row = (chat: ShellChat, inFolder = false) => (
    <div
      key={chat.id}
      className="ui-as-chat-row"
      data-indent={inFolder || undefined}
      draggable={renaming !== chat.id}
      data-dragging={drag?.id === chat.id || undefined}
      data-drop={
        over?.id === chat.id && drag?.kind === "chat" && drag.id !== chat.id
          ? over.after
            ? "after"
            : "before"
          : undefined
      }
      data-menu={menu?.id === chat.id || undefined}
      data-pinned={chat.pinned || undefined}
      onKeyDown={onKey(chat)}
      onDragStart={(e) => lift(e, chat.title, () => setDrag(chat.id))}
      onDragOver={(e) => {
        if (drag?.kind !== "chat") return;
        e.preventDefault();
        const r = e.currentTarget.getBoundingClientRect();
        const after = e.clientY > r.top + r.height / 2;
        if (over?.id !== chat.id || over.after !== after)
          setOver({ id: chat.id, after });
      }}
      onDrop={(e) => {
        e.preventDefault();
        if (over) drop(over.id, over.after);
        endDrag();
      }}
      onDragEnd={endDrag}
    >
      {renaming === chat.id ? (
        <input
          className="ui-as-chat-row__rename"
          defaultValue={chat.title}
          aria-label="Chat name"
          autoFocus
          onFocus={(e) => e.currentTarget.select()}
          onBlur={(e) => {
            const title = e.currentTarget.value.trim();
            if (title) update(chat.id, { title });
            setRenaming(null);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter") e.currentTarget.blur();
            if (e.key === "Escape") {
              e.currentTarget.value = chat.title;
              e.currentTarget.blur();
            }
          }}
        />
      ) : (
        <button
          type="button"
          className="ui-as-side__item ui-as-chat-row__link"
          aria-current={active === chat.title ? "page" : undefined}
          onClick={() => {
            if (chat.unread) update(chat.id, { unread: false });
            onOpen(chat);
          }}
        >
          <span className="ui-as-side__text" title={chat.title} dir="auto">
            {chat.title}
          </span>
          {chat.unread && (
            <span className="ui-as-chat-row__dot" aria-label="Unread" />
          )}
        </button>
      )}
      <span className="ui-as-chat-row__tools">
        <button
          type="button"
          className="ui-as-chat-row__tool"
          aria-label={`Options for ${chat.title}`}
          aria-haspopup="menu"
          aria-expanded={menu?.id === chat.id}
          data-chat-menu={chat.id}
          onClick={(e) => openMenu(chat.id, e.currentTarget)}
        >
          <Icon icon={MoreHorizontalIcon} size={15} />
        </button>
        <button
          type="button"
          className="ui-as-chat-row__tool ui-as-chat-row__pin"
          aria-label={chat.pinned ? `Unpin ${chat.title}` : `Pin ${chat.title}`}
          aria-pressed={!!chat.pinned}
          // One stop per row for the keyboard: pinning is also in the ⋯ menu and on ⌥⌘P
          tabIndex={-1}
          onClick={() => togglePin(chat)}
        >
          <Icon icon={chat.pinned ? PinOffIcon : PinIcon} size={14} />
        </button>
      </span>
    </div>
  );

  // A folder: opens and closes on a click, takes chats dropped on it, can be pinned and dragged into another place
  const folderRow = ({ name, pinned }: { name: string; pinned: boolean }) => {
    const inside = list.filter((c) => !c.pinned && c.folder === name);
    const isOpen = openFolders.has(name);
    const overMe =
      over?.id === `folder:${name}` &&
      drag?.kind === "folder" &&
      drag.id !== name;
    return (
      <Fragment key={name}>
        <div
          className="ui-as-chat-folder-row"
          draggable={renamingFolder !== name}
          data-menu={menu?.id === `folder:${name}` || undefined}
          data-dragging={
            (drag?.kind === "folder" && drag.id === name) || undefined
          }
          data-drop={overMe ? (over.after ? "after" : "before") : undefined}
          onDragStart={(e) =>
            lift(e, name, () => setDragState({ kind: "folder", id: name }))
          }
          onDragOver={(e) => {
            if (!drag) return;
            e.preventDefault();
            if (drag.kind === "chat") {
              setDropFolder(name);
              setOver(null);
              return;
            }
            const r = e.currentTarget.getBoundingClientRect();
            const after = e.clientY > r.top + r.height / 2;
            if (over?.id !== `folder:${name}` || over.after !== after)
              setOver({ id: `folder:${name}`, after });
          }}
          onDragLeave={() => setDropFolder((f) => (f === name ? null : f))}
          onDrop={(e) => {
            e.preventDefault();
            if (drag?.kind === "chat") dropInto(name);
            else if (drag?.kind === "folder" && over)
              moveFolder(drag.id, name, over.after);
            endDrag();
          }}
          onDragEnd={endDrag}
        >
          {renamingFolder === name ? (
            <input
              className="ui-as-chat-row__rename ui-as-chat-folder__rename"
              defaultValue={name}
              aria-label="Project name"
              autoFocus
              onFocus={(e) => e.currentTarget.select()}
              onBlur={(e) => renameFolder(name, e.currentTarget.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") e.currentTarget.blur();
                if (e.key === "Escape") {
                  e.currentTarget.value = name;
                  e.currentTarget.blur();
                }
              }}
            />
          ) : (
            <button
              type="button"
              className="ui-as-side__item ui-as-chat-folder"
              aria-expanded={isOpen}
              title={rail ? name : undefined}
              data-drop={dropFolder === name || undefined}
              onClick={() => toggleFolder(name)}
            >
              <Icon icon={isOpen ? FolderOpenIcon : Folder01Icon} size={16} />
              <span className="ui-as-side__text" title={name}>
                {name}
              </span>
              {/* On hover a small arrow right after the name: right when closed, down when open */}
              <span className="ui-as-chat-folder__arrow" aria-hidden="true">
                <Icon icon={ArrowRight01Icon} size={13} />
              </span>
            </button>
          )}
          <span className="ui-as-chat-row__tools">
            <button
              type="button"
              className="ui-as-chat-row__tool"
              aria-label={`Options for ${name}`}
              aria-haspopup="menu"
              aria-expanded={menu?.id === `folder:${name}`}
              data-chat-menu={`folder:${name}`}
              onClick={(e) => openMenu(`folder:${name}`, e.currentTarget)}
            >
              <Icon icon={MoreHorizontalIcon} size={15} />
            </button>
            <button
              type="button"
              className="ui-as-chat-row__tool ui-as-chat-row__pin"
              aria-label={pinned ? `Unpin ${name}` : `Pin ${name}`}
              aria-pressed={pinned}
              tabIndex={-1}
              onClick={() => togglePinFolder(name)}
            >
              <Icon icon={pinned ? PinOffIcon : PinIcon} size={14} />
            </button>
          </span>
        </div>
        {isOpen && (
          <div className="ui-as-chat-folder__chats">
            {inside.length === 0 ? (
              <span className="ui-as-side__empty ui-as-chat-folder__empty">
                No chats yet
              </span>
            ) : (
              inside.map((c) => row(c, true))
            )}
          </div>
        )}
      </Fragment>
    );
  };

  const pinnedChats = list.filter((c) => c.pinned);
  const pinnedFolders = folders.filter((f) => f.pinned);
  const otherFolders = folders.filter((f) => !f.pinned);
  const loose = list.filter(
    (c) => !c.pinned && !(c.folder && folderNames.includes(c.folder)),
  );
  const open = menu && list.find((c) => c.id === menu.id);
  const openFolder = menu?.id.startsWith("folder:")
    ? folders.find((f) => `folder:${f.name}` === menu.id)
    : undefined;
  return (
    <>
      {/* Pinned folders and chats on top, then the other folders with their chats indented inside, then the loose chats */}
      {(pinnedFolders.length > 0 || pinnedChats.length > 0) && (
        <div
          className="ui-as-side__group ui-as-chat-pinned"
          role="group"
          aria-label="Pinned"
        >
          <span className="ui-as-side__label">Pinned</span>
          {pinnedFolders.map(folderRow)}
          {pinnedChats.map((c) => row(c))}
        </div>
      )}
      <div
        className="ui-as-side__group ui-as-chat-folders"
        role="group"
        aria-label="Projects"
      >
        <span className="ui-as-side__label">Projects</span>
        {otherFolders.length === 0 && (
          <span className="ui-as-side__empty">No projects yet</span>
        )}
        {otherFolders.map(folderRow)}
      </div>
      {showLoose && (
        <div
          className="ui-as-side__group ui-as-chats"
          role="group"
          aria-label="Chats"
        >
          <span className="ui-as-side__label">Chats</span>
          {loose.length === 0 && (
            <span className="ui-as-side__empty">No chats yet</span>
          )}
          {loose.map((c) => row(c))}
        </div>
      )}
      {menu &&
        (open || openFolder) &&
        side.current &&
        createPortal(
          <div
            ref={panel}
            className="ui-as-menu ui-as-chat-menu"
            style={{ left: menu.left }}
            role="dialog"
            aria-label={`Options for ${open ? open.title : openFolder?.name}`}
            onKeyDown={(e) => {
              // Tab leaves the menu from its ⋯ button, so the focus goes on to the row's next control
              if (e.key !== "Tab") return;
              menuButton(menu.id)?.focus({ preventScroll: true });
              setMenu(null);
            }}
          >
            <MenuList
              items={
                open
                  ? itemsFor(open)
                  : openFolder
                    ? folderItems(openFolder.name, openFolder.pinned)
                    : []
              }
              onPick={() => {
                const button = menuButton(menu.id);
                setMenu(null);
                refocus(button);
              }}
            />
          </div>,
          side.current,
        )}
      {toast && app.current && createPortal(toast, app.current)}
    </>
  );
}

/* ---------- File cards ---------- */

export type FileType = "DOC" | "PDF" | "XLS";
export type FileItem = { name: string; type: FileType; edited?: string };

/** Badge color and label per file type */
const colors: Record<FileType, "blue" | "rose" | "emerald"> = {
  DOC: "blue",
  PDF: "rose",
  XLS: "emerald",
};
export const fileKind: Record<FileType, string> = {
  DOC: "Document",
  PDF: "PDF",
  XLS: "Spreadsheet",
};

/** A file as a card: preview on a gray stage, name and one line of details below */
export function FileCard({
  file,
  meta,
  onOpen,
}: {
  file: FileItem;
  meta: ReactNode;
  /** Called when the card is clicked or pressed */ onOpen?: (
    file: FileItem,
  ) => void;
}) {
  return (
    <button
      type="button"
      className="ui-as-template"
      title={file.name}
      onClick={onOpen && (() => onOpen(file))}
    >
      <span className="ui-as-template__preview">
        <DocPreview type={file.type} color={colors[file.type]} />
      </span>
      <b className="ui-as-template__name">{file.name}</b>
      <span className="ui-as-meta">{meta}</span>
    </button>
  );
}

/** The lines on each kind of page: one row per array, widths in % */
const docRows: Record<FileType, number[][]> = {
  DOC: [[100], [31, 31, 31], [48, 48], [31, 31, 31], [24, 48, 24], [31, 31]],
  PDF: [[100], [48, 48], [31, 62]],
  XLS: [
    [31, 31, 31],
    [31, 31, 31],
    [31, 31, 31],
    [31, 31, 31],
    [31, 31, 31],
  ],
};

/**
 * File illustration: a small white page with hairline rows of content, its top right corner cut off,
 * and a colored tag with the type overlapping its lower left corner. On hover the page lifts a little.
 */
function DocPreview({
  type,
  color,
}: {
  type: FileType;
  color: "blue" | "rose" | "emerald";
}) {
  return (
    <span className="ui-as-doc" data-color={color} aria-hidden="true">
      <span className="ui-as-doc__art">
        <span className="ui-as-doc__page" data-type={type}>
          {type === "PDF" && <span className="ui-as-doc__image" />}
          <span className="ui-as-doc__rows">
            {docRows[type].map((row, r) => (
              <span key={r} className="ui-as-doc__row">
                {row.map((w, k) => (
                  <i key={k} style={{ width: `${w}%` }} />
                ))}
              </span>
            ))}
          </span>
        </span>
        <span className="ui-as-doc__type">{type}</span>
      </span>
    </span>
  );
}

/* ---------- Workspace: the work area's tabs, like Codex – the screen's own tabs, artifacts and New tabs ---------- */

/**
 * A tab in the work area. Either the screen's own content – a task, a brief, a run's details – or an artifact by its
 * file name, shown in the shared viewers (document, sheet, folder, code, image) with its content from `artifacts`
 */
export type WorkAreaTab =
  | {
      id: string;
      /** Its name – the tab shows it, × and screen readers say it */
      title: string;
      /** What the tab shows instead of the plain title (the title still names it) */
      label?: ReactNode;
      icon?: IconSvgElement;
      /** A small number after the name, e.g. what waits on you */
      count?: number;
      content: ReactNode;
      artifact?: undefined;
    }
  | {
      /** Defaults to the file name */
      id?: string;
      /** The file name – its content comes from `artifacts` */
      artifact: string;
      /** Where it's from, the line under its name – e.g. the task or the run */
      from?: string;
    };

export type WorkAreaOptions = {
  /** The screen's own tabs and the artifacts it starts with – in this order */
  tabs?: WorkAreaTab[];
  /**
   * The ids of the tabs that are open when the screen starts – and again when the work area opens after its last tab
   * was closed. "new" adds a New tab. Default: every tab in `tabs`, or one New tab when there are none
   */
  defaultTabs?: string[];
  /** The tabs open when the screen starts, if they differ from defaultTabs – e.g. the task and two artifacts, while only the task comes back */
  initialTabs?: string[];
  /** What each artifact shows when it's opened, by file name */
  artifacts?: Record<string, ShellArtifact>;
  /** Called when an artifact is opened – it opens in its own tab, or its tab comes to the front */
  onOpenArtifact?: (name: string, from?: string) => void;
  /**
   * Called when you change an artifact in its tab – a cell of a sheet, its widths or styles, the text of a document,
   * a folder's files – with the whole artifact as it is now. The workspace keeps the change itself while the screen
   * lives; save it here
   */
  onArtifactChange?: (name: string, artifact: ShellArtifact) => void;
  /**
   * What a New tab (the + at the end of the tabs) shows: the start options – an address field, Tools and Suggested.
   * A suggestion with an `artifact` opens it in that tab. false: no + and no New tabs
   */
  newTab?: Pick<WorkspacePanelProps, "tools" | "sources" | "onOpen"> | false;
  /** While the page's data loads: the artifacts' tabs show a gray name and a gray file in their place – the tabs stay where they are */
  loading?: boolean;
};

/** One open tab */
type OpenTab = { key: string; page?: string; artifact?: string; from?: string };
type PageTab = Extract<WorkAreaTab, { content: ReactNode }>;
type WorkAreaState = {
  uid: string;
  loading: boolean;
  tabs: OpenTab[];
  active: string | null;
  /** Counts up whenever something asks to be seen – the shell opens the work area for it */
  reveal: number;
  page: (id: string) => PageTab | undefined;
  newTab: WorkAreaOptions["newTab"];
  select: (key: string) => void;
  close: (key: string) => void;
  /** The tabs the screen starts with, again */
  reset: () => void;
  openNew: () => string;
  /** A New tab turns into an artifact – or hands over to the artifact's tab when it's open already */
  openIn: (key: string, name: string, from?: string) => void;
  edited: (name: string) => boolean;
  memo: (name: string) => unknown;
  setMemo: (name: string, memo: unknown) => void;
  change: (name: string, next: ShellArtifact) => void;
  seed: (name: string, artifact: ShellArtifact) => void;
};

/** The work area's tabs, from useWorkArea – pass it to AppShell as `workspace` */
export type WorkArea = {
  /** Opens an artifact in its own tab (or brings its tab to the front) and opens the work area if it's folded away */
  open: (name: string, from?: string) => void;
  /** Brings one of the screen's own tabs back and to the front */
  show: (id: string) => void;
  /** Opens a New tab */
  openNew: () => void;
  /** An artifact as it is now – with your changes */
  artifact: (name: string) => ShellArtifact | undefined;
  /** For the App Shell */
  readonly state: WorkAreaState;
};

const tabIdOf = (t: WorkAreaTab) =>
  t.artifact !== undefined ? (t.id ?? t.artifact) : t.id;
const isWorkArea = (w: unknown): w is WorkArea =>
  !!w && typeof w === "object" && "state" in w && "open" in w;

/**
 * The work area's tabs and everything in them: which tabs are open, which one is in front, and what you changed in
 * each artifact – kept by file name while the screen lives, so it outlives its tab. Call it in your screen and pass
 * it to AppShell as `workspace`; open artifacts from anywhere in the page with `open(name)`
 */
export function useWorkArea(options: WorkAreaOptions = {}): WorkArea {
  const {
    tabs: pages = [],
    defaultTabs,
    initialTabs,
    artifacts = {},
    onOpenArtifact,
    onArtifactChange,
    newTab = {},
    loading = false,
  } = options;
  const uid = useId().replace(/:/g, "");
  const fresh = useRef(0);
  const defaultsOf = (start = false): OpenTab[] => {
    const ids =
      (start ? initialTabs : undefined) ??
      defaultTabs ??
      (pages.length ? pages.map(tabIdOf) : ["new"]);
    return ids.flatMap((id): OpenTab[] => {
      if (id === "new")
        return newTab === false ? [] : [{ key: `n:${++fresh.current}` }];
      const t = pages.find((x) => tabIdOf(x) === id);
      if (!t) return [];
      return t.artifact !== undefined
        ? [{ key: `a:${t.artifact}`, artifact: t.artifact, from: t.from }]
        : [{ key: `p:${t.id}`, page: t.id }];
    });
  };
  const [s, setS] = useState<{ tabs: OpenTab[]; active: string | null }>(() => {
    const tabs = defaultsOf(true);
    return { tabs, active: tabs[0]?.key ?? null };
  });
  const [reveal, setReveal] = useState(0);
  // What you changed in an artifact – by file name, so it outlives its tab: switch away, close it, open it again
  const [edits, setEdits] = useState<Record<string, ShellArtifact>>({});
  // Where you were in one – the sheet's selection, the folder's path, view and sort
  const [memos, setMemos] = useState<Record<string, unknown>>({});
  // Content for names that had none: a renamed or duplicated file, a subfolder opened in its own tab
  const [seeds, setSeeds] = useState<Record<string, ShellArtifact>>({});
  const artifact = (name: string) =>
    edits[name] ?? seeds[name] ?? artifacts[name];

  const open = (name: string, from?: string) => {
    onOpenArtifact?.(name, from);
    const key = `a:${name}`;
    setS((x) => {
      // Opened after the last tab was closed: the screen's own tabs come back with it
      const base = x.tabs.length ? x.tabs : defaultsOf();
      return {
        tabs: base.some((t) => t.key === key)
          ? base
          : [...base, { key, artifact: name, from }],
        active: key,
      };
    });
    setReveal((n) => n + 1);
  };
  const show = (id: string) => {
    const key = `p:${id}`;
    setS((x) => ({
      tabs: x.tabs.some((t) => t.key === key)
        ? x.tabs
        : [...x.tabs, { key, page: id }],
      active: key,
    }));
    setReveal((n) => n + 1);
  };
  const openNew = () => {
    const key = `n:${++fresh.current}`;
    setS((x) => ({ tabs: [...x.tabs, { key }], active: key }));
    setReveal((n) => n + 1);
    return key;
  };
  const state: WorkAreaState = {
    uid,
    loading,
    tabs: s.tabs,
    active: s.tabs.some((t) => t.key === s.active)
      ? s.active
      : (s.tabs[0]?.key ?? null),
    reveal,
    page: (id) =>
      pages.find((p): p is PageTab => p.artifact === undefined && p.id === id),
    newTab,
    select: (key) => setS((x) => ({ ...x, active: key })),
    close: (key) =>
      setS((x) => {
        const at = x.tabs.findIndex((t) => t.key === key);
        if (at < 0) return x;
        const rest = x.tabs.filter((t) => t.key !== key);
        // Its neighbour comes to the front – the one after it, or the one before at the end
        return {
          tabs: rest,
          active:
            x.active === key
              ? (rest[Math.min(at, rest.length - 1)]?.key ?? null)
              : x.active,
        };
      }),
    reset: () =>
      setS(() => {
        const tabs = defaultsOf();
        return { tabs, active: tabs[0]?.key ?? null };
      }),
    openNew,
    openIn: (from, name, origin) => {
      onOpenArtifact?.(name, origin);
      const key = `a:${name}`;
      setS((x) =>
        x.tabs.some((t) => t.key === key)
          ? { tabs: x.tabs.filter((t) => t.key !== from), active: key }
          : {
              tabs: x.tabs.map((t) =>
                t.key === from ? { key, artifact: name, from: origin } : t,
              ),
              active: key,
            },
      );
    },
    edited: (name) => name in edits,
    memo: (name) => memos[name],
    setMemo: (name, m) => setMemos((x) => ({ ...x, [name]: m })),
    change: (name, next) => {
      setEdits((e) => ({ ...e, [name]: next }));
      onArtifactChange?.(name, next);
    },
    seed: (name, a) =>
      setSeeds((x) =>
        x[name] || edits[name] || artifacts[name] ? x : { ...x, [name]: a },
      ),
  };
  return { open, show, openNew: () => void openNew(), artifact, state };
}

const WorkAreaContext = createContext<WorkArea | null>(null);
/** The work area's tabs from inside the page – e.g. to open an artifact from a chip deep in a message: useShellWorkArea()?.open(name) */
export const useShellWorkArea = () => useContext(WorkAreaContext);

/** What a tab shows: its glyph, name and count */
const tabFace = (
  s: WorkAreaState,
  t: OpenTab,
): { title: string; label?: ReactNode; glyph: ReactNode; count?: number } => {
  if (t.artifact !== undefined)
    return s.loading
      ? {
          title: "Loading",
          label: <Skeleton width={72} height={8} />,
          glyph: <Skeleton width={14} height={14} radius={4} />,
        }
      : {
          title: t.artifact,
          glyph: <Icon icon={fileOutline(t.artifact)} size={14} />,
        };
  if (t.page !== undefined) {
    const p = s.page(t.page);
    return {
      title: p?.title ?? t.page,
      label: p?.label,
      glyph: <Icon icon={p?.icon ?? DashboardSquare02Icon} size={14} />,
      count: p?.count,
    };
  }
  return { title: "New tab", glyph: <Icon icon={Globe02Icon} size={14} /> };
};

/**
 * The tabs across the work area's head, like Codex's: every tab with its own ×, a + at the end for a New tab. More tabs
 * first narrow, their names fading out, then the whole strip scrolls sideways. ← → Home End move, Delete closes the
 * focused tab; closing the last one folds the work area away
 */
function WorkAreaTabs({ ws, label }: { ws: WorkArea; label: string }) {
  const s = ws.state;
  const list = useRef<HTMLDivElement>(null);
  const keys = s.tabs.map((t) => t.key);
  const active = s.active ?? "";
  const focus = (key: string) =>
    window.setTimeout(() =>
      list.current
        ?.querySelector<HTMLElement>(`[data-tab="${CSS.escape(key)}"]`)
        ?.focus({ preventScroll: true }),
    );
  const close = (key: string, refocus: boolean) => {
    const at = keys.indexOf(key);
    const rest = keys.filter((k) => k !== key);
    const next = key === active ? rest[Math.min(at, rest.length - 1)] : active;
    s.close(key);
    // The focus goes to the tab that comes to the front; with none left the shell moves it to its open button
    if (refocus && next) focus(next);
  };
  // A newly opened tab scrolls into sight in the strip – never the page
  useEffect(() => {
    const strip = list.current;
    const tab = strip?.querySelector<HTMLElement>(
      `[data-tab="${CSS.escape(active)}"]`,
    );
    if (!strip || !tab) return;
    const box = (
      tab.closest<HTMLElement>(".ui-asw-ptab-wrap") ?? tab
    ).getBoundingClientRect();
    const view = strip.getBoundingClientRect();
    if (box.left < view.left) strip.scrollLeft -= view.left - box.left + 8;
    else if (box.right > view.right - 16)
      strip.scrollLeft += box.right - view.right + 24;
  }, [active, keys.length]);
  // The end of the strip fades out only while more tabs wait behind it
  const [more, setMore] = useState(false);
  useLayoutEffect(() => {
    const strip = list.current;
    if (!strip) return;
    const measure = () =>
      setMore(strip.scrollWidth - strip.clientWidth - strip.scrollLeft > 1);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(strip);
    strip.addEventListener("scroll", measure, { passive: true });
    return () => {
      ro.disconnect();
      strip.removeEventListener("scroll", measure);
    };
  }, [keys.length]);
  return (
    <div className="ui-asw-bar">
      <div
        ref={list}
        className="ui-asw-ptabs"
        role="tablist"
        aria-label={label}
        data-more={more || undefined}
        onKeyDown={(e) => {
          const at =
            (document.activeElement as HTMLElement | null)?.dataset.tab ??
            active;
          if (e.key === "Delete" || e.key === "Backspace") {
            e.preventDefault();
            return close(at, true);
          }
          rove(e, keys, at, (key) => {
            s.select(key);
            focus(key);
          });
        }}
      >
        {s.tabs.map((t, i) => {
          const face = tabFace(s, t);
          const on = t.key === active;
          return (
            <span
              key={t.key}
              className="ui-asw-ptab-wrap"
              data-on={on || undefined}
            >
              <button
                type="button"
                role="tab"
                id={`${s.uid}-tab-${i}`}
                aria-controls={on ? `${s.uid}-pane-${i}` : undefined}
                className="ui-asw-ptab"
                data-tab={t.key}
                aria-selected={on}
                aria-keyshortcuts="Delete"
                tabIndex={on ? 0 : -1}
                title={face.title}
                onClick={() => s.select(t.key)}
              >
                {face.glyph}
                <span className="ui-asw-ptab__name" dir="auto">
                  {face.label ?? face.title}
                </span>
                {!!face.count && (
                  <span className="ui-asw-ptab__count">
                    {shortCount(face.count)}
                  </span>
                )}
              </button>
              <button
                type="button"
                className="ui-asw-ptab__close"
                tabIndex={-1}
                aria-label={`Close ${face.title}`}
                onClick={() => close(t.key, false)}
              >
                <Icon icon={Cancel01Icon} size={10} stroke={2.2} />
              </button>
            </span>
          );
        })}
      </div>
      {s.newTab !== false && (
        <button
          type="button"
          className="ui-asw-ptab-new"
          aria-label="New tab"
          title="New tab"
          onClick={(e) => {
            const key = s.openNew();
            // From the keyboard the focus follows to the new tab; a click leaves it where it is
            if (e.detail === 0) focus(key);
          }}
        >
          <Icon icon={PlusSignIcon} size={14} />
        </button>
      )}
    </div>
  );
}

/** An artifact while the page loads: its head (glyph, name, when) and gray lines where the file goes */
function ArtifactSkeleton() {
  return (
    <Loading label="Loading the file" className="ui-as-art-skel">
      <div className="ui-as-art-skel__head" data-skel="">
        <Skeleton width={24} height={24} radius={6} />
        <span className="ui-as-art-skel__name">
          <Skeleton width={132} height={10} />
          <Skeleton width={56} height={8} />
        </span>
      </div>
      <div className="ui-as-art-skel__body" data-skel="">
        {[100, 92, 96, 64, 0, 100, 88, 72].map((w, i) =>
          w ? (
            <Skeleton key={i} width={`${w}%`} height={10} />
          ) : (
            <span key={i} className="ui-as-art-skel__gap" />
          ),
        )}
      </div>
    </Loading>
  );
}

/** What's in the tabs: the screen's own stay mounted while they're open, artifacts and New tabs render in front only */
function WorkAreaBody({ ws }: { ws: WorkArea }) {
  const s = ws.state;
  const nt = s.newTab || {};
  return (
    <>
      {s.tabs.map((t, i) => {
        const on = t.key === s.active;
        if (t.page === undefined && !on) return null;
        return (
          <div
            key={t.key}
            className="ui-asw-pane"
            role="tabpanel"
            id={`${s.uid}-pane-${i}`}
            aria-labelledby={`${s.uid}-tab-${i}`}
            hidden={!on}
          >
            {t.page !== undefined ? (
              s.page(t.page)?.content
            ) : t.artifact !== undefined && s.loading ? (
              <ArtifactSkeleton />
            ) : t.artifact !== undefined ? (
              <ArtifactView
                key={t.artifact}
                file={{ name: t.artifact, from: t.from }}
                artifact={ws.artifact(t.artifact)}
                edited={s.edited(t.artifact)}
                lookup={ws.artifact}
                memo={s.memo(t.artifact)}
                onMemo={(m) => s.setMemo(t.artifact!, m)}
                onChange={(next) => s.change(t.artifact!, next)}
                onSeed={s.seed}
                onOpen={(name) => ws.open(name, t.from)}
              />
            ) : (
              <WorkspacePanel
                tools={nt.tools}
                sources={nt.sources}
                onOpen={(id) => {
                  const source = (nt.sources ?? defaultSources).find(
                    (x) => x.id === id,
                  );
                  if (source?.artifact)
                    s.openIn(t.key, source.artifact, source.source);
                  nt.onOpen?.(id);
                }}
              />
            )}
          </div>
        );
      })}
    </>
  );
}

/**
 * One opened artifact in its tab: its name and where it's from, then what's in it – a document to write in, a sheet
 * to work in, code, an image or a folder to browse. What you change goes up through onChange, so it outlives the tab
 */
function ArtifactView({
  file,
  artifact,
  edited,
  lookup,
  memo,
  onMemo,
  onChange,
  onOpen,
  onSeed,
}: {
  file: { name: string; from?: string };
  artifact?: ShellArtifact;
  edited: boolean;
  /** Another artifact by its name – for a folder's Quick Look and its subfolders */
  lookup: (name: string) => ShellArtifact | undefined;
  /** Where you were in it – the sheet's selection, the folder's path and view – kept while the screen lives */
  memo?: unknown;
  onMemo: (memo: unknown) => void;
  onChange: (next: ShellArtifact) => void;
  onOpen: (name: string) => void;
  /** Gives a name content when it has none yet – a renamed or duplicated file, a subfolder opened in its own tab */
  onSeed: (name: string, artifact: ShellArtifact) => void;
}) {
  const a = artifact ?? {};
  const ref = useRef<HTMLDivElement>(null);
  const width = useWidth(ref);
  const sub = [file.from, a.updated].filter(Boolean);
  return (
    <div ref={ref} className="ui-asw-art">
      <header className="ui-asw-art__head">
        <span className="ui-asw-art__glyph" aria-hidden="true">
          {fileGlyph(file.name)}
        </span>
        <span className="ui-asw-art__title">
          <b dir="auto" title={file.name}>
            {file.name}
          </b>
          {sub.length > 0 && (
            <span>
              {sub.map((x, i) => (
                <Fragment key={i}>
                  {i > 0 && <i aria-hidden="true"> · </i>}
                  {x}
                </Fragment>
              ))}
            </span>
          )}
        </span>
      </header>
      {a.preview ??
        (a.doc ? (
          <DocEditor
            name={file.name}
            doc={a.doc}
            edited={edited}
            width={width}
            onChange={(html) => onChange({ ...a, doc: { ...a.doc!, html } })}
          />
        ) : a.table ? (
          <SheetView
            name={file.name}
            table={a.table}
            meta={a.sheet}
            width={width}
            memo={memo as SheetMemo | undefined}
            onMemo={onMemo}
            onChange={(table, sheet) => onChange({ ...a, table, sheet })}
          />
        ) : a.code ? (
          <StillView kind="code" code={a.code} name={file.name} />
        ) : a.image ? (
          <StillView kind="image" image={a.image} name={file.name} />
        ) : a.files ? (
          <FolderView
            name={file.name}
            entries={a.files}
            lookup={lookup}
            width={width}
            memo={memo as FolderMemo | undefined}
            onMemo={onMemo}
            onOpen={onOpen}
            onSeed={onSeed}
            onChange={(files) =>
              onChange({
                ...a,
                files,
                updated:
                  files.length === 1 ? "1 item" : `${files.length} items`,
              })
            }
          />
        ) : (
          <p className="ui-asw-art__none">No preview for this file yet.</p>
        ))}
    </div>
  );
}

/** Artifacts as soft gray file chips – a small two-gray glyph by type; each opens its artifact in the work area */
export function ArtifactChips({
  names,
  onOpen,
}: {
  names: string[];
  onOpen: (name: string) => void;
}) {
  return (
    <div className="ui-asw-files">
      <style href="ui-app-shell-artifacts" precedence="default">
        {artifactCss}
      </style>
      {names.map((f) => (
        <button
          key={f}
          type="button"
          className="ui-asw-file"
          title={f}
          onClick={() => onOpen(f)}
        >
          {fileGlyph(f)}
          <span className="ui-asw-file__name">{middle(f)}</span>
        </button>
      ))}
    </div>
  );
}

const number0 = new Intl.NumberFormat("en");

/* ---------- Artifacts in the workspace: types and the shared viewers ---------- */

/** One gray block of an image drawn as a wireframe, on a 120 × 80 grid */
export type ShellBlock = {
  x: number;
  y: number;
  w: number;
  h: number;
  /** Corner radius (default 2) */ r?: number;
  tone?: "ink" | "mid" | "soft";
};

/** One entry of a folder: just its name, or with what Finder shows about it */
export type ShellFolderEntry =
  | string
  | {
      name: string;
      /** In bytes – or in words, e.g. "12 KB" */
      size?: number | string;
      /** When it last changed – an ISO date or a timestamp in ms */
      modified?: string | number;
      /** e.g. "YAML" – made up from the extension when left out; "Folder" makes it a folder */
      kind?: string;
      /** A subfolder's own entries – or give the subfolder an artifact of its own, by its name */
      files?: ShellFolderEntry[];
    };
/** How a sheet's cell shows a number */
export type ShellNumberFormat = "general" | "number" | "percent" | "currency";
export type ShellCellStyle = { bold?: boolean; format?: ShellNumberFormat };

/** What an opened artifact shows in its own tab of the Task panel – one of these, by the file's kind */
export type ShellArtifact = {
  /** e.g. "Updated 2h ago" */
  updated?: string;
  /**
   * A document: a title, a few paragraphs and points – points that start with "[ ] " or "[x] " make a checklist.
   * It opens as a page you write in; what you wrote comes back as `html`, which then replaces the rest
   */
  doc?: {
    title: string;
    paragraphs?: string[];
    bullets?: string[];
    html?: string;
  };
  /** A sheet: the first row is the header; a cell that starts with = is a formula */
  table?: string[][];
  /** The sheet's column widths in px and its cell styles by "row:column" (from 0) – set as you work in it */
  sheet?: { widths?: number[]; styles?: Record<string, ShellCellStyle> };
  /** Code or data, shown as is */
  code?: string;
  /** An image, as gray blocks on a 120 × 80 grid – or pass `preview` */
  image?: ShellBlock[];
  /** A folder: what's in it – a name without an extension is a subfolder */
  files?: ShellFolderEntry[];
  /** Your own preview – replaces all of the above */
  preview?: ReactNode;
};

/* File chips: soft gray, a small two-gray icon by type – like the attachments in Collaboration Stream */
const middle = (label: string, max = 30) => {
  if (label.length <= max) return label;
  const tail = Math.min(12, Math.floor(max / 2.5));
  return `${label.slice(0, max - tail - 1).trimEnd()}…${label.slice(-tail)}`;
};
/** doc · sheet · code · folder · image – each glyph carries its type, and the type its color (--asw-file-*) */
type GlyphType = "doc" | "sheet" | "code" | "folder" | "image";
const FileGlyph = ({ type = "doc" }: { type?: GlyphType }) => (
  <svg
    className="ui-asw-glyph"
    data-type={type}
    width="12"
    height="12"
    viewBox="0 0 12 12"
    fill="none"
    aria-hidden="true"
  >
    <path
      d="M3.3 1h3.9L10 3.8v6.4c0 .72-.58 1.3-1.3 1.3H3.3c-.72 0-1.3-.58-1.3-1.3V2.3C2 1.58 2.58 1 3.3 1Z"
      fill="currentColor"
      opacity="0.45"
    />
    <path d="M7.2 1v1.8c0 .55.45 1 1 1H10Z" fill="currentColor" />
    <path
      d="M4 6.6h4M4 8.6h2.6"
      stroke="currentColor"
      strokeWidth="1.1"
      strokeLinecap="round"
    />
  </svg>
);
const TableGlyph = () => (
  <svg
    className="ui-asw-glyph"
    data-type="sheet"
    width="12"
    height="12"
    viewBox="0 0 12 12"
    fill="none"
    aria-hidden="true"
  >
    <path
      d="M3.3 1h3.9L10 3.8v6.4c0 .72-.58 1.3-1.3 1.3H3.3c-.72 0-1.3-.58-1.3-1.3V2.3C2 1.58 2.58 1 3.3 1Z"
      fill="currentColor"
      opacity="0.45"
    />
    <path d="M7.2 1v1.8c0 .55.45 1 1 1H10Z" fill="currentColor" />
    <rect
      x="3.4"
      y="5.6"
      width="2.3"
      height="1.6"
      rx="0.4"
      fill="currentColor"
    />
    <rect
      x="6.3"
      y="5.6"
      width="2.3"
      height="1.6"
      rx="0.4"
      fill="currentColor"
    />
    <rect
      x="3.4"
      y="7.8"
      width="2.3"
      height="1.6"
      rx="0.4"
      fill="currentColor"
    />
    <rect
      x="6.3"
      y="7.8"
      width="2.3"
      height="1.6"
      rx="0.4"
      fill="currentColor"
    />
  </svg>
);
const FolderGlyph = () => (
  <svg
    className="ui-asw-glyph"
    data-type="folder"
    width="12"
    height="12"
    viewBox="0 0 12 12"
    fill="none"
    aria-hidden="true"
  >
    <path
      d="M1.4 3.1c0-.72.58-1.3 1.3-1.3h1.72c.38 0 .73.19.94.5l.6.9h3.34c.72 0 1.3.58 1.3 1.3V6H1.4Z"
      fill="currentColor"
    />
    <path
      d="M1 6.95C.86 6.3 1.36 5.7 2.02 5.7h7.96c.66 0 1.16.6 1.02 1.25l-.5 2.4a1.3 1.3 0 0 1-1.27 1.05H2.77A1.3 1.3 0 0 1 1.5 9.35Z"
      fill="currentColor"
      opacity="0.45"
    />
  </svg>
);
/** Sheets get the table, a name without an extension the folder, everything else the page */
/** The tabs keep it quiet: a gray outline icon by type – the colored two-tone glyphs are for the content below */
const fileOutline = (name: string): IconSvgElement => {
  const ext = name.includes(".")
    ? name.slice(name.lastIndexOf(".") + 1).toLowerCase()
    : "";
  if (!ext) return Folder01Icon;
  if (["csv", "xls", "xlsx", "ods", "tsv"].includes(ext)) return Table01Icon;
  if (["png", "jpg", "jpeg", "gif", "webp", "svg", "avif"].includes(ext))
    return Image01Icon;
  if (
    [
      "json",
      "yaml",
      "yml",
      "js",
      "ts",
      "tsx",
      "jsx",
      "py",
      "sh",
      "toml",
      "sql",
      "css",
      "html",
    ].includes(ext)
  )
    return FileCodeIcon;
  return File01Icon;
};

export const fileGlyph = (name: string) => {
  const ext = name.includes(".")
    ? name.slice(name.lastIndexOf(".") + 1).toLowerCase()
    : "";
  if (!ext) return <FolderGlyph />;
  if (["csv", "xls", "xlsx", "ods", "tsv"].includes(ext)) return <TableGlyph />;
  if (["png", "jpg", "jpeg", "gif", "webp", "svg", "avif"].includes(ext))
    return <FileGlyph type="image" />;
  if (
    [
      "json",
      "yaml",
      "yml",
      "js",
      "ts",
      "tsx",
      "jsx",
      "py",
      "sh",
      "toml",
      "sql",
      "css",
      "html",
    ].includes(ext)
  )
    return <FileGlyph type="code" />;
  return <FileGlyph />;
};

export function Wireframe({ blocks }: { blocks: ShellBlock[] }) {
  return (
    <svg
      className="ui-asw-wire"
      viewBox="0 0 120 80"
      aria-hidden="true"
      preserveAspectRatio="xMidYMid meet"
    >
      <rect
        className="ui-asw-wire__frame"
        x="0.5"
        y="0.5"
        width="119"
        height="79"
        rx="6"
      />
      {blocks.map((b, i) => (
        <rect
          key={i}
          className="ui-asw-wire__block"
          data-tone={b.tone ?? "soft"}
          x={b.x}
          y={b.y}
          width={b.w}
          height={b.h}
          rx={b.r ?? 2}
        />
      ))}
    </svg>
  );
}

function useWidth(ref: RefObject<HTMLElement | null>) {
  const [w, setW] = useState(0);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const measure = () => setW(el.clientWidth);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

/** The nearest ancestor that scrolls up and down – the panel's body */
const scrollerOf = (el: HTMLElement | null) => {
  for (let n = el?.parentElement; n; n = n.parentElement)
    if (/(auto|scroll)/.test(getComputedStyle(n).overflowY)) return n;
  return null;
};
/** Scrolls the panel just enough to show the element – never sideways */
const revealY = (el: HTMLElement | null) => {
  const box = scrollerOf(el);
  if (!el || !box) return;
  const b = box.getBoundingClientRect();
  const r = el.getBoundingClientRect();
  if (r.top < b.top + 4) box.scrollTop -= b.top + 4 - r.top;
  else if (r.bottom > b.bottom - 4) box.scrollTop += r.bottom - b.bottom + 4;
};

/* ---------- Toolbars: one Tab stop, ← → between the tools, a tip under each ---------- */

const mac =
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
const MOD = mac ? "⌘" : "Ctrl+";
const SHIFT = mac ? "⇧" : "Shift+";
/** A tool keeps the focus where you work – the sheet or the page – when it's clicked */
const keep = (e: { preventDefault: () => void }) => e.preventDefault();

/** The tools of a toolbar that are on screen – not the items of a menu that's open from it */
const toolsOf = (bar: HTMLElement) =>
  [...bar.querySelectorAll<HTMLElement>("[data-tool]")].filter(
    (el) => !el.closest(".ui-as-popover") && el.getClientRects().length > 0,
  );

function Toolbar({
  label,
  className,
  children,
}: {
  label: string;
  className?: string;
  children: ReactNode;
}) {
  const bar = useRef<HTMLDivElement>(null);
  const stop = useRef(0);
  const tipEl = useRef<HTMLSpanElement>(null);
  const wait = useRef(0);
  const [tip, setTip] = useState<{
    text: string;
    keys?: string;
    x: number;
    y: number;
  } | null>(null);
  // Exactly one tool takes Tab – the last one you were on
  const sync = () => {
    const el = bar.current;
    if (!el) return;
    const tools = toolsOf(el);
    stop.current = clamp(stop.current, 0, Math.max(0, tools.length - 1));
    el.querySelectorAll<HTMLElement>("[data-tool]").forEach(
      (t) => (t.tabIndex = -1),
    );
    if (tools[stop.current]) tools[stop.current].tabIndex = 0;
  };
  useLayoutEffect(sync);
  useEffect(() => () => window.clearTimeout(wait.current), []);
  // The tip stays inside the toolbar's width, so it never widens the panel
  useLayoutEffect(() => {
    const t = tipEl.current;
    const el = bar.current;
    if (!t || !el || !tip) return;
    t.style.left = `${clamp(tip.x - t.offsetWidth / 2, 4, el.clientWidth - t.offsetWidth - 4)}px`;
  }, [tip]);
  const show = (t: HTMLElement) => {
    const el = bar.current;
    if (!el || !t.dataset.tip || t.getAttribute("aria-expanded") === "true")
      return setTip(null);
    const b = t.getBoundingClientRect();
    const h = el.getBoundingClientRect();
    setTip({
      text: t.dataset.tip,
      keys: t.dataset.keys,
      x: b.left - h.left + b.width / 2,
      y: b.bottom - h.top + 6,
    });
  };
  const hide = () => {
    window.clearTimeout(wait.current);
    setTip(null);
  };
  return (
    <div
      ref={bar}
      role="toolbar"
      aria-label={label}
      className={["ui-asw-tb", className].filter(Boolean).join(" ")}
      onKeyDown={(e) => {
        const t = e.target as HTMLElement;
        if (t.closest(".ui-as-popover")) return;
        if (e.key === "Escape") return hide();
        const tools = toolsOf(bar.current!);
        const at = tools.indexOf(t);
        const step =
          e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
        const to = step
          ? (at + step + tools.length) % tools.length
          : e.key === "Home"
            ? 0
            : e.key === "End"
              ? tools.length - 1
              : -1;
        if (at < 0 || to < 0) return;
        e.preventDefault();
        stop.current = to;
        sync();
        tools[to].focus();
      }}
      onFocus={(e) => {
        const t = e.target as HTMLElement;
        const at = toolsOf(bar.current!).indexOf(t);
        if (at >= 0) {
          stop.current = at;
          sync();
        }
        if (at >= 0 && t.matches(":focus-visible")) show(t);
        else hide();
      }}
      onBlur={hide}
      onPointerOver={(e) => {
        const t = (e.target as HTMLElement).closest<HTMLElement>("[data-tip]");
        if (!t || t.closest(".ui-as-popover")) return;
        window.clearTimeout(wait.current);
        wait.current = window.setTimeout(() => show(t), tip ? 0 : 480);
      }}
      onPointerLeave={hide}
      onPointerDown={hide}
    >
      {children}
      {tip && (
        <span
          ref={tipEl}
          className="ui-asw-tip"
          aria-hidden="true"
          style={{ top: tip.y }}
        >
          {tip.text}
          {tip.keys && <kbd>{tip.keys}</kbd>}
        </span>
      )}
    </div>
  );
}

/** A 28px icon button of a toolbar – its name in a tip, with the shortcut */
function Tool({
  icon,
  label,
  keys,
  pressed,
  disabled,
  onClick,
}: {
  icon: IconSvgElement;
  label: string;
  keys?: string;
  pressed?: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="ui-asw-tool"
      data-tool
      data-tip={label}
      data-keys={keys}
      aria-label={label}
      aria-pressed={pressed}
      aria-disabled={disabled || undefined}
      onMouseDown={keep}
      onClick={() => !disabled && onClick()}
    >
      <Icon icon={icon} size={16} />
    </button>
  );
}

/** A tool that opens a menu – with what's chosen as text when there's room */
function MenuTool({
  icon,
  label,
  text,
  items,
  align = "start",
}: {
  icon: IconSvgElement;
  label: string;
  text?: string;
  items: MenuItem[];
  align?: "start" | "end";
}) {
  return (
    <Dropdown
      align={align}
      items={items}
      className="ui-asw-tool-anchor"
      trigger={({ open, toggle }) => (
        <button
          type="button"
          className="ui-asw-tool ui-asw-tool--menu"
          data-tool
          data-tip={label}
          aria-label={text ? `${label}: ${text}` : label}
          aria-haspopup="menu"
          aria-expanded={open}
          onMouseDown={keep}
          onClick={toggle}
        >
          <Icon icon={icon} size={16} />
          {text && <span className="ui-asw-tool__text">{text}</span>}
          <Icon icon={ArrowDown01Icon} size={12} />
        </button>
      )}
    />
  );
}

/* ---------- Context menus: right-click anywhere in an artifact – the shell's menu look, at the pointer ---------- */

/** One line of a context menu – every one with an icon; the shortcut shows on the right */
type CtxItem = {
  label: string;
  icon: IconSvgElement;
  keys?: string;
  disabled?: boolean;
  danger?: boolean;
  /** A line above it */
  divider?: boolean;
  /** A choice – shows a check when true */
  checked?: boolean;
  submenu?: CtxItem[];
  onSelect?: () => void;
};
type CtxAt<K extends string> = {
  x: number;
  y: number;
  kind: K;
  back: HTMLElement | null;
};
/** Shift F10 and the context menu key open it from the keyboard */
const isMenuKey = (e: { key: string; shiftKey: boolean }) =>
  e.key === "ContextMenu" || (e.shiftKey && e.key === "F10");

/**
 * A context menu for one viewer: open it at the pointer (or on an element, from the keyboard) with a kind, and render
 * it with items made from the viewer's state right now – so every action works on what's selected at that moment
 */
function useContextMenu<K extends string>() {
  const [at, setAt] = useState<CtxAt<K> | null>(null);
  const open = (x: number, y: number, kind: K, back?: HTMLElement | null) =>
    setAt({
      x,
      y,
      kind,
      back: back ?? (document.activeElement as HTMLElement | null),
    });
  const openOn = (
    el: Element | null | undefined,
    kind: K,
    back?: HTMLElement | null,
  ) => {
    const r = el?.getBoundingClientRect();
    if (r)
      open(
        r.left + Math.min(12, r.width / 2),
        r.top + Math.min(r.height, 26),
        kind,
        back,
      );
  };
  const render = (build: (kind: K) => { label: string; items: CtxItem[] }) => {
    if (!at) return null;
    const { label, items } = build(at.kind);
    return (
      <ContextMenu
        key={`${at.x}:${at.y}:${at.kind}`}
        x={at.x}
        y={at.y}
        label={label}
        items={items}
        back={at.back}
        onClose={(refocus) => {
          setAt(null);
          if (refocus && at.back?.isConnected)
            at.back.focus({ preventScroll: true });
        }}
      />
    );
  };
  return { open, openOn, render, isOpen: !!at };
}

/** The menu itself: at the pointer, flipped to stay on screen; a click outside, a scroll or Esc closes it */
function ContextMenu({
  x,
  y,
  label,
  items,
  back,
  onClose,
}: {
  x: number;
  y: number;
  label: string;
  items: CtxItem[];
  back: HTMLElement | null;
  onClose: (refocus: boolean) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const close = useRef(onClose);
  close.current = onClose;
  const host = back?.closest<HTMLElement>(".ui-as") ?? document.body;
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    // A transformed ancestor moves "fixed" – measure where it landed and correct by that
    const r = el.getBoundingClientRect();
    const ox = r.left - x;
    const oy = r.top - y;
    const left =
      x + r.width > window.innerWidth - 6 ? Math.max(6, x - r.width) : x;
    const top =
      y + r.height > window.innerHeight - 6 ? Math.max(6, y - r.height) : y;
    el.style.left = `${left - ox}px`;
    el.style.top = `${top - oy}px`;
  }, [x, y]);
  useEffect(() => {
    const inside = (t: EventTarget | null) =>
      !!t && !!ref.current?.contains(t as Node);
    const down = (e: PointerEvent) => !inside(e.target) && close.current(false);
    const scroll = (e: Event) => !inside(e.target) && close.current(false);
    const away = () => close.current(false);
    document.addEventListener("pointerdown", down, true);
    window.addEventListener("scroll", scroll, true);
    window.addEventListener("resize", away);
    window.addEventListener("blur", away);
    return () => {
      document.removeEventListener("pointerdown", down, true);
      window.removeEventListener("scroll", scroll, true);
      window.removeEventListener("resize", away);
      window.removeEventListener("blur", away);
    };
  }, []);
  return createPortal(
    // Portaled, but React still bubbles its events up the component tree – they stop here
    <div
      ref={ref}
      className="ui-as-menu ui-asw-ctx"
      onContextMenu={(e) => {
        e.preventDefault();
        e.stopPropagation();
      }}
      onPointerDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      onDoubleClick={(e) => e.stopPropagation()}
    >
      <CtxPanel
        items={items}
        label={label}
        close={(refocus) => close.current(refocus)}
        focusFirst
      />
    </div>,
    host,
  );
}

/** One level of a context menu: ↑ ↓ Home End move, → or Enter opens a submenu, ← or Esc goes back, Tab closes */
function CtxPanel({
  items,
  label,
  close,
  onBack,
  focusFirst,
}: {
  items: CtxItem[];
  label: string;
  close: (refocus: boolean) => void;
  onBack?: () => void;
  focusFirst?: boolean;
}) {
  const list = useRef<HTMLDivElement>(null);
  const [sub, setSub] = useState<{ at: number; key: boolean } | null>(null);
  const wait = useRef(0);
  const rows = () => [
    ...(list.current?.querySelectorAll<HTMLElement>(
      ":scope > .ui-asw-ctx__row > [role^='menuitem']",
    ) ?? []),
  ];
  useLayoutEffect(() => {
    // A submenu that would leave the screen opens to the other side, and moves up when it's too low
    const panel = list.current?.parentElement;
    if (panel?.classList.contains("ui-asw-ctx__sub")) {
      const r = panel.getBoundingClientRect();
      if (r.right > window.innerWidth - 6) panel.dataset.flip = "";
      if (r.bottom > window.innerHeight - 6)
        panel.style.top = `${window.innerHeight - 6 - r.bottom - 4}px`;
    }
    if (focusFirst)
      (
        rows().find((b) => !b.hasAttribute("aria-disabled")) ?? rows()[0]
      )?.focus({ preventScroll: true });
    return () => window.clearTimeout(wait.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const pick = (i: number, key: boolean) => {
    const it = items[i];
    if (!it || it.disabled) return;
    if (it.submenu) return setSub({ at: i, key });
    close(true);
    it.onSelect?.();
  };
  return (
    <div
      ref={list}
      role="menu"
      aria-label={label}
      className="ui-as-menu__list"
      onKeyDown={(e) => {
        e.stopPropagation();
        const bs = rows();
        const at = bs.indexOf(document.activeElement as HTMLElement);
        const go = (i: number) => {
          e.preventDefault();
          setSub(null);
          bs[(i + bs.length) % bs.length]?.focus({ preventScroll: true });
        };
        if (e.key === "ArrowDown") go(at + 1);
        else if (e.key === "ArrowUp") go(at < 0 ? -1 : at - 1);
        else if (e.key === "Home") go(0);
        else if (e.key === "End") go(-1);
        else if (
          (e.key === "ArrowRight" && items[at]?.submenu) ||
          ((e.key === "Enter" || e.key === " ") && at >= 0)
        ) {
          e.preventDefault();
          pick(at, true);
        } else if (e.key === "ArrowLeft" && onBack) {
          e.preventDefault();
          onBack();
        } else if (e.key === "Escape") {
          e.preventDefault();
          if (onBack) onBack();
          else close(true);
        } else if (e.key === "Tab") {
          e.preventDefault();
          close(true);
        }
      }}
    >
      {items.map((it, i) => (
        <Fragment key={`${i}-${it.label}`}>
          {it.divider && i > 0 && (
            <span className="ui-as-menu__sep" role="separator" />
          )}
          <span className="ui-asw-ctx__row">
            <button
              type="button"
              role={it.checked === undefined ? "menuitem" : "menuitemcheckbox"}
              aria-checked={it.checked}
              aria-disabled={it.disabled || undefined}
              aria-haspopup={it.submenu ? "menu" : undefined}
              aria-expanded={it.submenu ? sub?.at === i : undefined}
              tabIndex={-1}
              className="ui-as-menu__item"
              data-danger={it.danger || undefined}
              data-open={(it.submenu && sub?.at === i) || undefined}
              onPointerEnter={(e) => {
                if (e.pointerType === "mouse")
                  e.currentTarget.focus({ preventScroll: true });
                window.clearTimeout(wait.current);
                wait.current = window.setTimeout(
                  () =>
                    setSub(
                      it.submenu && !it.disabled ? { at: i, key: false } : null,
                    ),
                  it.submenu ? 90 : 180,
                );
              }}
              onClick={() => pick(i, false)}
            >
              <Icon icon={it.icon} size={15} />
              <span className="ui-as-menu__label">
                <span className="ui-as-menu__text">{it.label}</span>
              </span>
              {it.keys && <kbd className="ui-as-menu__kbd">{it.keys}</kbd>}
              {it.checked && <Icon icon={Tick02Icon} size={14} />}
              {it.submenu && <Icon icon={ArrowRight01Icon} size={14} />}
            </button>
            {it.submenu && sub?.at === i && (
              <span className="ui-as-menu ui-asw-ctx__sub">
                <CtxPanel
                  items={it.submenu}
                  label={it.label}
                  close={close}
                  focusFirst={sub.key}
                  onBack={() => {
                    setSub(null);
                    rows()[i]?.focus({ preventScroll: true });
                  }}
                />
              </span>
            )}
          </span>
        </Fragment>
      ))}
    </div>
  );
}

/** Text to the clipboard – quietly, where the browser doesn't allow it */
const toClipboard = (text: string) => {
  try {
    navigator.clipboard?.writeText(text).catch(() => {});
  } catch {
    /* nothing to do */
  }
};
/** Text from the clipboard, or null where the browser keeps it to itself */
const fromClipboard = async () => {
  try {
    return await navigator.clipboard.readText();
  } catch {
    return null;
  }
};

/** Right-click on code or an image: Copy, and Select all for code */
function StillView({
  kind,
  code,
  image,
  name,
}: {
  kind: "code" | "image";
  code?: string;
  image?: ShellBlock[];
  name: string;
}) {
  const ref = useRef<HTMLElement>(null);
  const ctx = useContextMenu<"still">();
  const [said, setSaid] = useState("");
  const selectAll = () => {
    const el = ref.current;
    const s = window.getSelection();
    if (!el || !s) return;
    const r = document.createRange();
    r.selectNodeContents(el);
    s.removeAllRanges();
    s.addRange(r);
  };
  const copy = () => {
    const s = window.getSelection();
    const picked =
      s && !s.isCollapsed && ref.current?.contains(s.anchorNode)
        ? s.toString()
        : "";
    toClipboard(
      kind === "code"
        ? picked || code || ""
        : (ref.current?.querySelector("svg")?.outerHTML ?? ""),
    );
    setSaid("");
    window.setTimeout(
      () =>
        setSaid(
          picked
            ? "Copied the selection"
            : kind === "code"
              ? "Copied the code"
              : "Copied the image",
        ),
      30,
    );
  };
  const menu = (e: {
    preventDefault: () => void;
    clientX: number;
    clientY: number;
  }) => {
    e.preventDefault();
    ctx.open(e.clientX, e.clientY, "still", ref.current);
  };
  const keys = (e: ReactKeyboardEvent<HTMLElement>) => {
    if (!isMenuKey(e)) return;
    e.preventDefault();
    ctx.openOn(ref.current, "still", ref.current);
  };
  return (
    <>
      {kind === "code" ? (
        <pre
          ref={ref as RefObject<HTMLPreElement>}
          className="ui-asw-art__code"
          tabIndex={0}
          aria-label={`${name}, code`}
          onContextMenu={menu}
          onKeyDown={keys}
        >
          {code}
        </pre>
      ) : (
        <div
          ref={ref as RefObject<HTMLDivElement>}
          className="ui-asw-art__image"
          tabIndex={0}
          role="img"
          aria-label={`${name}, image`}
          onContextMenu={menu}
          onKeyDown={keys}
        >
          <Wireframe blocks={image ?? []} />
        </div>
      )}
      {ctx.render(() => ({
        label: name,
        items: [
          {
            label: kind === "code" ? "Copy" : "Copy image",
            icon: Copy01Icon,
            keys: `${MOD}C`,
            onSelect: copy,
          },
          ...(kind === "code"
            ? [
                {
                  label: "Select all",
                  icon: CursorTextIcon,
                  keys: `${MOD}A`,
                  onSelect: selectAll,
                },
              ]
            : []),
        ],
      }))}
      <span className="ui-asw-sr" role="status" aria-live="polite">
        {said}
      </span>
    </>
  );
}

/* ---------- Document: a small Word ---------- */

type DocBlock = "p" | "h1" | "h2" | "h3" | "blockquote";
type DocState = {
  block: DocBlock;
  bold: boolean;
  italic: boolean;
  underline: boolean;
  strike: boolean;
  code: boolean;
  link: boolean;
  ul: boolean;
  ol: boolean;
  check: boolean;
};
const DOC_BLOCKS: { id: DocBlock; label: string; icon: IconSvgElement }[] = [
  { id: "p", label: "Text", icon: TextIcon },
  { id: "h1", label: "Heading 1", icon: Heading01Icon },
  { id: "h2", label: "Heading 2", icon: Heading02Icon },
  { id: "h3", label: "Heading 3", icon: Heading03Icon },
  { id: "blockquote", label: "Quote", icon: QuoteDownIcon },
];
const DOC_IDLE: DocState = {
  block: "p",
  bold: false,
  italic: false,
  underline: false,
  strike: false,
  code: false,
  link: false,
  ul: false,
  ol: false,
  check: false,
};
const esc = (s: string) =>
  s.replace(
    /[&<>"]/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!,
  );
/** A document as HTML – what you wrote in it, or its title, paragraphs and points; points that start with "[ ] " or "[x] " are a checklist */
const docHtml = (d: NonNullable<ShellArtifact["doc"]>) => {
  if (d.html !== undefined) return d.html;
  const points = d.bullets ?? [];
  const check =
    points.length > 0 && points.every((b) => /^\[( |x)\] /i.test(b));
  const list = points.length
    ? check
      ? `<ul data-check>${points.map((b) => `<li${/^\[x\]/i.test(b) ? " data-done" : ""}>${esc(b.slice(4))}</li>`).join("")}</ul>`
      : `<ul>${points.map((b) => `<li>${esc(b)}</li>`).join("")}</ul>`
    : "";
  return [
    `<h1>${esc(d.title)}</h1>`,
    ...(d.paragraphs ?? []).map((p) => `<p>${esc(p)}</p>`),
    list,
  ].join("");
};
/** Markdown as you type: what the line starts with, then a space */
const MARKS: Record<string, DocBlock | "ul" | "ol" | "check"> = {
  "#": "h1",
  "##": "h2",
  "###": "h3",
  ">": "blockquote",
  "-": "ul",
  "*": "ul",
  "1.": "ol",
  "[ ]": "check",
  "[]": "check",
};

/**
 * The document as a page you write in, like Word: a compact toolbar – undo, the block style, bold, italic, underline,
 * strikethrough, code, a link, three kinds of lists and clear – that shows what the selection is; Markdown as you type
 * ("# ", "- ", "1. ", "[ ] "); pastes come in as plain text. Under it the words, the characters and whether you edited it
 */
function DocEditor({
  name,
  doc,
  edited,
  width,
  onChange,
}: {
  name: string;
  doc: NonNullable<ShellArtifact["doc"]>;
  edited: boolean;
  width: number;
  onChange: (html: string) => void;
}) {
  const ed = useRef<HTMLDivElement>(null);
  const saved = useRef<Range | null>(null);
  const [st, setSt] = useState<DocState>(DOC_IDLE);
  const [count, setCount] = useState({ words: 0, chars: 0 });
  const [link, setLink] = useState<{
    url: string;
    had: boolean;
    fromKey: boolean;
  } | null>(null);
  const [said, setSaid] = useState("");
  const changeRef = useRef(onChange);
  changeRef.current = onChange;

  const measure = () => {
    const t = ed.current?.innerText ?? "";
    setCount({
      words: (t.match(/\S+/g) ?? []).length,
      chars: t.replace(/\n/g, "").length,
    });
  };
  // Filled once – from then on the page is yours, React never writes into it again
  useLayoutEffect(() => {
    if (!ed.current) return;
    ed.current.innerHTML = docHtml(doc);
    measure();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** The element the caret is in, if it's inside the page and matches */
  const near = (sel: string) => {
    const el = ed.current;
    const s = window.getSelection();
    const n = s?.anchorNode;
    const at = n instanceof Element ? n : n?.parentElement;
    const hit = at?.closest<HTMLElement>(sel);
    return el && hit && el.contains(hit) && hit !== el ? hit : null;
  };
  // The toolbar follows the selection
  useEffect(() => {
    const read = () => {
      const el = ed.current;
      const s = window.getSelection();
      if (!el || !s?.rangeCount || !el.contains(s.anchorNode)) return;
      saved.current = s.getRangeAt(0).cloneRange();
      const blk = near("h1, h2, h3, blockquote");
      const list = near("ul, ol");
      setSt({
        block: (blk?.tagName.toLowerCase() as DocBlock | undefined) ?? "p",
        bold: document.queryCommandState("bold"),
        italic: document.queryCommandState("italic"),
        underline: document.queryCommandState("underline"),
        strike: document.queryCommandState("strikeThrough"),
        code: !!near("code"),
        link: !!near("a"),
        ul: list?.tagName === "UL" && !list.hasAttribute("data-check"),
        ol: list?.tagName === "OL",
        check: !!list?.hasAttribute("data-check"),
      });
    };
    document.addEventListener("selectionchange", read);
    return () => document.removeEventListener("selectionchange", read);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const announce = (text: string) => {
    setSaid("");
    window.setTimeout(() => setSaid(text), 30);
  };
  /** Back into the page with the selection you had – when the focus was on the toolbar or in a menu */
  const restore = () => {
    const el = ed.current;
    if (!el) return;
    const s = window.getSelection();
    if (
      document.activeElement === el &&
      s?.rangeCount &&
      el.contains(s.anchorNode)
    )
      return;
    el.focus({ preventScroll: true });
    const r = saved.current;
    if (r && el.contains(r.startContainer) && s) {
      s.removeAllRanges();
      s.addRange(r);
    }
  };
  const changed = () => {
    if (!ed.current) return;
    measure();
    changeRef.current(ed.current.innerHTML);
    document.dispatchEvent(new Event("selectionchange"));
  };
  const run = (cmd: string, value?: string) => {
    restore();
    document.execCommand(cmd, false, value);
    changed();
  };
  const block = (b: DocBlock) => {
    restore();
    if (near("li") && b !== "p")
      document.execCommand(
        near("ol") ? "insertOrderedList" : "insertUnorderedList",
      );
    document.execCommand(
      "formatBlock",
      false,
      `<${b === st.block && b !== "p" ? "p" : b}>`,
    );
    changed();
  };
  const list = (kind: "ul" | "ol" | "check") => {
    restore();
    const cur = near("ul, ol");
    const isCheck = !!cur?.hasAttribute("data-check");
    if (kind === "check") {
      if (isCheck) document.execCommand("insertUnorderedList");
      else {
        if (cur?.tagName !== "UL") document.execCommand("insertUnorderedList");
        near("ul")?.setAttribute("data-check", "");
      }
    } else if (kind === "ul" && isCheck && cur) {
      cur.removeAttribute("data-check");
      cur
        .querySelectorAll(":scope > li[data-done]")
        .forEach((li) => li.removeAttribute("data-done"));
    } else {
      document.execCommand(
        kind === "ul" ? "insertUnorderedList" : "insertOrderedList",
      );
      near("ol")?.removeAttribute("data-check");
    }
    changed();
  };
  const code = () => {
    restore();
    const c = near("code");
    const s = window.getSelection();
    if (!s) return;
    if (c) {
      const r = document.createRange();
      r.selectNode(c);
      s.removeAllRanges();
      s.addRange(r);
      document.execCommand("insertText", false, c.textContent ?? "");
    } else if (!s.isCollapsed)
      document.execCommand(
        "insertHTML",
        false,
        `<code>${esc(s.toString())}</code>`,
      );
    else return announce("Select some text first");
    changed();
  };
  const clear = () => {
    restore();
    const s = window.getSelection();
    document.execCommand("removeFormat");
    document.execCommand("unlink");
    // Code isn't a format to the browser – take it out by hand
    const r = s?.rangeCount ? s.getRangeAt(0) : null;
    ed.current?.querySelectorAll("code").forEach((c) => {
      if (r && (r.intersectsNode(c) || c.contains(r.startContainer)))
        c.replaceWith(document.createTextNode(c.textContent ?? ""));
    });
    changed();
    announce("Formatting cleared");
  };
  const toggleDone = (li: HTMLElement) => {
    const on = !li.hasAttribute("data-done");
    li.toggleAttribute("data-done", on);
    changed();
    announce(on ? "Checked" : "Unchecked");
  };
  const openLink = (fromKey: boolean) => {
    if (!fromKey) restore();
    const a = near("a");
    setLink({ url: a?.getAttribute("href") ?? "", had: !!a, fromKey });
  };
  const applyLink = (url: string) => {
    const u = url.trim();
    setLink(null);
    restore();
    const s = window.getSelection();
    const a = near("a");
    if (a && s) {
      const r = document.createRange();
      r.selectNodeContents(a);
      s.removeAllRanges();
      s.addRange(r);
    }
    if (!u) document.execCommand("unlink");
    else {
      const href = /^([a-z][a-z0-9+.-]*:|\/|#)/i.test(u) ? u : `https://${u}`;
      if (s?.isCollapsed)
        document.execCommand(
          "insertHTML",
          false,
          `<a href="${esc(href)}">${esc(u)}</a>`,
        );
      else document.execCommand("createLink", false, href);
    }
    changed();
  };
  /** "# " at the start of a line makes a heading, "- " a list … – the marks themselves go */
  const markdown = () => {
    const el = ed.current;
    const s = window.getSelection();
    if (!el || !s?.isCollapsed || !s.anchorNode || near("li")) return;
    let top: Node = s.anchorNode;
    while (top.parentNode && top.parentNode !== el) top = top.parentNode;
    const blk = near("p, div, h1, h2, h3, blockquote");
    const r = document.createRange();
    if (blk) r.setStart(blk, 0);
    else r.setStartBefore(top);
    r.setEnd(s.anchorNode, s.anchorOffset);
    const before = r.toString().replace(/ /g, " ");
    const to = before.endsWith(" ") ? MARKS[before.slice(0, -1)] : undefined;
    if (!to) return;
    s.removeAllRanges();
    s.addRange(r);
    document.execCommand("delete");
    if (to === "ul" || to === "ol" || to === "check") list(to);
    else document.execCommand("formatBlock", false, `<${to}>`);
  };

  const onKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const mod = e.metaKey || e.ctrlKey;
    if (isMenuKey(e)) {
      e.preventDefault();
      const s = window.getSelection();
      const r = s?.rangeCount ? s.getRangeAt(0).getBoundingClientRect() : null;
      if (r && (r.width || r.height))
        ctx.open(r.left, r.bottom + 4, "doc", ed.current);
      else ctx.openOn(ed.current, "doc", ed.current);
      return;
    }
    if (mod && !e.altKey) {
      const k = e.key.toLowerCase();
      const act = !e.shiftKey
        ? (
            {
              b: () => run("bold"),
              i: () => run("italic"),
              u: () => run("underline"),
              k: () => openLink(true),
              e: code,
            } as Record<string, () => void>
          )[k]
        : (
            {
              Digit7: () => list("ol"),
              Digit8: () => list("ul"),
              Digit9: () => list("check"),
              KeyX: () => run("strikeThrough"),
            } as Record<string, () => void>
          )[e.code];
      if (act) {
        // Ours, not the shell's (⌘K is its search everywhere else)
        e.preventDefault();
        e.stopPropagation();
        return act();
      }
      if (e.key === "Enter") {
        const li = near("ul[data-check] > li");
        if (li) {
          e.preventDefault();
          toggleDone(li);
        }
      }
      return;
    }
    // Tab nests a list item, Shift Tab takes it back out – outside a list Tab moves on as usual
    if (e.key === "Tab" && near("li")) {
      e.preventDefault();
      const wasCheck = !!near("ul[data-check]");
      document.execCommand(e.shiftKey ? "outdent" : "indent");
      if (wasCheck) near("ul")?.setAttribute("data-check", "");
      changed();
    }
  };

  /* Right-click: on a word that isn't selected, the word gets selected first – like Word and Pages */
  const ctx = useContextMenu<"doc">();
  const wordAt = (x: number, y: number) => {
    const s = window.getSelection();
    if (!s) return;
    const inSel =
      !s.isCollapsed &&
      s.rangeCount > 0 &&
      [...s.getRangeAt(0).getClientRects()].some(
        (r) => x >= r.left && x <= r.right && y >= r.top && y <= r.bottom,
      );
    if (inSel) return;
    const r = document.caretRangeFromPoint?.(x, y);
    if (!r || !ed.current?.contains(r.startContainer)) return;
    s.removeAllRanges();
    s.addRange(r);
    if (
      /\S/.test(r.startContainer.textContent?.[r.startOffset] ?? "") ||
      /\S/.test(r.startContainer.textContent?.[r.startOffset - 1] ?? "")
    ) {
      s.modify("move", "forward", "character");
      s.modify("move", "backward", "word");
      s.modify("extend", "forward", "word");
    }
  };
  const paste = async () => {
    const text = await fromClipboard();
    if (text === null)
      return announce(
        `Press ${MOD}V to paste – the browser keeps the clipboard to itself here`,
      );
    restore();
    document.execCommand("insertText", false, text);
    changed();
  };
  const turnText = () => {
    if (st.ul || st.ol || st.check)
      list(st.check ? "check" : st.ol ? "ol" : "ul");
    else block("p");
  };
  const docMenu = (): CtxItem[] => {
    const s = window.getSelection();
    const picked =
      !!s && !s.isCollapsed && !!ed.current?.contains(s.anchorNode);
    const inList = st.ul || st.ol || st.check;
    return [
      {
        label: "Cut",
        icon: Scissor01Icon,
        keys: `${MOD}X`,
        disabled: !picked,
        onSelect: () => run("cut"),
      },
      {
        label: "Copy",
        icon: Copy01Icon,
        keys: `${MOD}C`,
        disabled: !picked,
        onSelect: () => run("copy"),
      },
      { label: "Paste", icon: ClipboardIcon, keys: `${MOD}V`, onSelect: paste },
      // Paste always comes in as plain text – one entry is enough; the styles wait one level down
      {
        label: "Format",
        icon: TextFontIcon,
        divider: true,
        submenu: [
          {
            label: "Bold",
            icon: TextBoldIcon,
            keys: `${MOD}B`,
            checked: st.bold,
            onSelect: () => run("bold"),
          },
          {
            label: "Italic",
            icon: TextItalicIcon,
            keys: `${MOD}I`,
            checked: st.italic,
            onSelect: () => run("italic"),
          },
          {
            label: "Underline",
            icon: TextUnderlineIcon,
            keys: `${MOD}U`,
            checked: st.underline,
            onSelect: () => run("underline"),
          },
        ],
      },
      {
        label: "Turn into",
        icon: TextSelectionIcon,
        submenu: [
          ...DOC_BLOCKS.map((b) => ({
            label: b.label,
            icon: b.icon,
            checked: !inList && st.block === b.id,
            onSelect: () => (b.id === "p" ? turnText() : block(b.id)),
          })),
          {
            label: "Bulleted list",
            icon: LeftToRightListBulletIcon,
            keys: `${SHIFT}${MOD}8`,
            checked: st.ul,
            divider: true,
            onSelect: () => !st.ul && list("ul"),
          },
          {
            label: "Numbered list",
            icon: LeftToRightListNumberIcon,
            keys: `${SHIFT}${MOD}7`,
            checked: st.ol,
            onSelect: () => !st.ol && list("ol"),
          },
          {
            label: "Checklist",
            icon: CheckListIcon,
            keys: `${SHIFT}${MOD}9`,
            checked: st.check,
            onSelect: () => !st.check && list("check"),
          },
        ],
      },
      {
        label: "Link…",
        icon: Link01Icon,
        keys: `${MOD}K`,
        divider: true,
        onSelect: () => openLink(true),
      },
      {
        label: "Select all",
        icon: CursorTextIcon,
        keys: `${MOD}A`,
        onSelect: () => run("selectAll"),
      },
    ];
  };

  // Tools that don't fit fold into "More" – the page stays one calm line of tools
  const roomy = width >= 470;
  const showBlockText = width >= 330;
  const blockNow = DOC_BLOCKS.find((b) => b.id === st.block) ?? DOC_BLOCKS[0];
  const more: MenuItem[] = [
    ...(roomy
      ? []
      : [
          {
            label: "Bulleted list",
            icon: LeftToRightListBulletIcon,
            shortcut: `${SHIFT}${MOD}8`,
            onSelect: () => list("ul"),
          },
          {
            label: "Numbered list",
            icon: LeftToRightListNumberIcon,
            shortcut: `${SHIFT}${MOD}7`,
            onSelect: () => list("ol"),
          },
          {
            label: "Checklist",
            icon: CheckListIcon,
            shortcut: `${SHIFT}${MOD}9`,
            onSelect: () => list("check"),
          },
        ]),
    {
      label: "Strikethrough",
      icon: TextStrikethroughIcon,
      shortcut: `${SHIFT}${MOD}X`,
      divider: !roomy,
      onSelect: () => run("strikeThrough"),
    },
    {
      label: "Inline code",
      icon: SourceCodeIcon,
      shortcut: `${MOD}E`,
      onSelect: code,
    },
    {
      label: "Clear formatting",
      icon: RemoveFormattingIcon,
      divider: true,
      onSelect: clear,
    },
  ];
  const plural = (n: number, one: string) =>
    `${number0.format(n)} ${one}${n === 1 ? "" : "s"}`;

  return (
    <div className="ui-asw-dc">
      <Toolbar label="Formatting" className="ui-asw-dc__tools">
        <span className="ui-asw-tb__group">
          <Tool
            icon={Undo02Icon}
            label="Undo"
            keys={`${MOD}Z`}
            onClick={() => run("undo")}
          />
          <Tool
            icon={Redo02Icon}
            label="Redo"
            keys={`${SHIFT}${MOD}Z`}
            onClick={() => run("redo")}
          />
        </span>
        <span className="ui-asw-tb__group">
          <MenuTool
            icon={blockNow.icon}
            label="Block style"
            text={showBlockText ? blockNow.label : undefined}
            items={DOC_BLOCKS.map((b) => ({
              label: b.label,
              icon: b.icon,
              checked: b.id === st.block,
              onSelect: () => block(b.id),
            }))}
          />
        </span>
        <span className="ui-asw-tb__group">
          <Tool
            icon={TextBoldIcon}
            label="Bold"
            keys={`${MOD}B`}
            pressed={st.bold}
            onClick={() => run("bold")}
          />
          <Tool
            icon={TextItalicIcon}
            label="Italic"
            keys={`${MOD}I`}
            pressed={st.italic}
            onClick={() => run("italic")}
          />
          <Tool
            icon={TextUnderlineIcon}
            label="Underline"
            keys={`${MOD}U`}
            pressed={st.underline}
            onClick={() => run("underline")}
          />
          <span className="ui-as-anchor">
            <Tool
              icon={Link01Icon}
              label="Link"
              keys={`${MOD}K`}
              pressed={st.link}
              onClick={() => (link ? setLink(null) : openLink(false))}
            />
            <Popover
              open={!!link}
              onClose={() => {
                const back = link?.fromKey;
                setLink(null);
                if (back) window.setTimeout(restore);
              }}
              className="ui-asw-linkpop"
            >
              {link && (
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    applyLink(
                      new FormData(e.currentTarget).get("url")?.toString() ??
                        "",
                    );
                  }}
                >
                  <input
                    name="url"
                    type="text"
                    inputMode="url"
                    autoFocus
                    defaultValue={link.url}
                    placeholder="Paste or type a link"
                    aria-label="Link address"
                    spellCheck={false}
                  />
                  {link.had && (
                    <button
                      type="button"
                      className="ui-asw-btn ui-asw-btn--quiet"
                      onClick={() => applyLink("")}
                    >
                      Remove
                    </button>
                  )}
                  <button type="submit" className="ui-asw-btn">
                    Apply
                  </button>
                </form>
              )}
            </Popover>
          </span>
          {roomy && (
            <>
              <Tool
                icon={TextStrikethroughIcon}
                label="Strikethrough"
                keys={`${SHIFT}${MOD}X`}
                pressed={st.strike}
                onClick={() => run("strikeThrough")}
              />
              <Tool
                icon={SourceCodeIcon}
                label="Inline code"
                keys={`${MOD}E`}
                pressed={st.code}
                onClick={code}
              />
            </>
          )}
        </span>
        {roomy && (
          <span className="ui-asw-tb__group">
            <Tool
              icon={LeftToRightListBulletIcon}
              label="Bulleted list"
              keys={`${SHIFT}${MOD}8`}
              pressed={st.ul}
              onClick={() => list("ul")}
            />
            <Tool
              icon={LeftToRightListNumberIcon}
              label="Numbered list"
              keys={`${SHIFT}${MOD}7`}
              pressed={st.ol}
              onClick={() => list("ol")}
            />
            <Tool
              icon={CheckListIcon}
              label="Checklist"
              keys={`${SHIFT}${MOD}9`}
              pressed={st.check}
              onClick={() => list("check")}
            />
          </span>
        )}
        <span className="ui-asw-tb__group">
          {roomy ? (
            <Tool
              icon={RemoveFormattingIcon}
              label="Clear formatting"
              onClick={clear}
            />
          ) : (
            <MenuTool
              icon={MoreHorizontalIcon}
              label="More formatting"
              items={more}
              align="end"
            />
          )}
        </span>
      </Toolbar>
      <div
        ref={ed}
        className="ui-asw-doc"
        contentEditable
        suppressContentEditableWarning
        role="textbox"
        aria-multiline="true"
        aria-label={`${name}, document`}
        spellCheck
        onFocus={() =>
          document.execCommand("defaultParagraphSeparator", false, "p")
        }
        onKeyDown={onKey}
        onInput={(e) => {
          const ie = e.nativeEvent as InputEvent;
          if (ie.inputType === "insertText" && ie.data === " ") markdown();
          // A new item of a checklist starts unchecked
          if (ie.inputType === "insertParagraph")
            near("ul[data-check] > li")?.removeAttribute("data-done");
          changed();
        }}
        onPaste={(e) => {
          e.preventDefault();
          document.execCommand(
            "insertText",
            false,
            e.clipboardData.getData("text/plain"),
          );
        }}
        onContextMenu={(e) => {
          e.preventDefault();
          wordAt(e.clientX, e.clientY);
          ctx.open(e.clientX, e.clientY, "doc", ed.current);
        }}
        onPointerDown={(e) => {
          // The box in front of a checklist item ticks it
          const li = (e.target as HTMLElement).closest<HTMLElement>(
            "ul[data-check] > li",
          );
          if (!li) return;
          const box = li.getBoundingClientRect();
          if (e.clientX - box.left < 22 && e.clientY - box.top < 22) {
            e.preventDefault();
            toggleDone(li);
          }
        }}
      />
      {ctx.render(() => ({ label: "Edit", items: docMenu() }))}
      <p className="ui-asw-dc__foot ui-asw-num">
        {plural(count.words, "word")}
        <i aria-hidden="true"> · </i>
        {plural(count.chars, "character")}
        {edited && (
          <>
            <i aria-hidden="true"> · </i>Edited
          </>
        )}
      </p>
      <span className="ui-asw-sr" role="status" aria-live="polite">
        {said}
      </span>
    </div>
  );
}

/* ---------- Sheet: a small Excel ---------- */

type SheetModel = {
  data: string[][];
  style: (ShellCellStyle | undefined)[][];
  widths: number[];
};
/** The active cell (r, c) and the other corner of the selection (r2, c2) */
type Sel = { r: number; c: number; r2: number; c2: number };
type SheetMemo = { sel: Sel };
type Val = number | string | boolean | null | { err: string };
type FNode =
  | { k: "lit"; v: Val }
  | { k: "ref"; r: number; c: number }
  | { k: "range"; r1: number; c1: number; r2: number; c2: number }
  | { k: "fn"; name: string; args: FNode[] }
  | { k: "un"; op: string; a: FNode }
  | { k: "bin"; op: string; a: FNode; b: FNode };
type Tok = { t: string; v?: string | number; r?: number; c?: number };

const ROW_H = 26;
const HEAD_H = 22;
const NUM_W = 30;
const COL_W = 76;
const colName = (c: number) => {
  let s = "";
  for (let n = c + 1; n > 0; n = Math.floor((n - 1) / 26))
    s = String.fromCharCode(65 + ((n - 1) % 26)) + s;
  return s;
};
const colIndex = (s: string) =>
  [...s.toUpperCase()].reduce((n, ch) => n * 26 + ch.charCodeAt(0) - 64, 0) - 1;
const cellName = (r: number, c: number) => `${colName(c)}${r + 1}`;
const box = (s: Sel) => ({
  r1: Math.min(s.r, s.r2),
  r2: Math.max(s.r, s.r2),
  c1: Math.min(s.c, s.c2),
  c2: Math.max(s.c, s.c2),
});
const isErr = (v: unknown): v is { err: string } =>
  typeof v === "object" && v !== null && "err" in v;
const hasStyle = (s?: ShellCellStyle) =>
  !!s && (s.bold !== undefined || (!!s.format && s.format !== "general"));

/**
 * A number as it's typed: "5,000", "1.8%" (0.018, like Excel) – and a number with a short unit after it, "38 ms",
 * "1.1 GB", counts as that number, so SUM over a column of timings works. "20k a minute" is text
 */
const LEAD =
  /^\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+)\s*(%?)\s*([A-Za-zµ/]{0,4})\s*$/;
const leadNum = (s: string) => {
  const m = LEAD.exec(s);
  if (!m) return null;
  const n = parseFloat(m[1].replace(/,/g, ""));
  return m[2] ? n / 100 : n;
};
/** A number without a unit – only those take a number format */
const plainNum = (s: string) => {
  const m = LEAD.exec(s);
  return !!m && !m[3];
};

function tokenize(s: string): Tok[] {
  const out: Tok[] = [];
  let i = 0;
  while (i < s.length) {
    const rest = s.slice(i);
    let m: RegExpExecArray | null;
    if ((m = /^\s+/.exec(rest))) {
      /* space */
    } else if ((m = /^#REF!/i.exec(rest))) out.push({ t: "referr" });
    else if ((m = /^"((?:[^"]|"")*)"/.exec(rest)))
      out.push({ t: "str", v: m[1].replace(/""/g, '"') });
    else if ((m = /^\$?([A-Za-z]{1,3})\$?(\d+)(?![A-Za-z0-9_(])/.exec(rest)))
      out.push({ t: "ref", r: +m[2] - 1, c: colIndex(m[1]) });
    else if ((m = /^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/.exec(rest)))
      out.push({ t: "num", v: parseFloat(m[0]) });
    else if ((m = /^[A-Za-z_][A-Za-z0-9_.]*/.exec(rest)))
      out.push({ t: "id", v: m[0].toUpperCase() });
    else if ((m = /^(?:<=|>=|<>|[-+*/^&=<>%(),:])/.exec(rest)))
      out.push({ t: m[0] });
    else throw { err: "#NAME?" };
    i += m[0].length;
  }
  return out;
}

const CMP = ["=", "<>", "<", ">", "<=", ">="];
/** A formula (without its "=") as a tree: comparisons, &, + -, * /, ^, a unary minus, %, refs, ranges and functions */
function parseFormula(src: string): FNode {
  const toks = tokenize(src);
  let i = 0;
  const is = (t: string) => toks[i]?.t === t;
  const eat = (t: string) => {
    if (!is(t)) throw { err: "#NAME?" };
    return toks[i++];
  };
  const level = (next: () => FNode, ops: string[]) => () => {
    let a = next();
    while (toks[i] && ops.includes(toks[i].t)) {
      const op = toks[i++].t;
      a = { k: "bin", op, a, b: next() };
    }
    return a;
  };
  const primary = (): FNode => {
    const t = toks[i++];
    if (!t) throw { err: "#NAME?" };
    if (t.t === "num") return { k: "lit", v: t.v as number };
    if (t.t === "str") return { k: "lit", v: t.v as string };
    if (t.t === "referr") return { k: "lit", v: { err: "#REF!" } };
    if (t.t === "ref") {
      if (!is(":")) return { k: "ref", r: t.r!, c: t.c! };
      i++;
      const b = eat("ref");
      return {
        k: "range",
        r1: Math.min(t.r!, b.r!),
        c1: Math.min(t.c!, b.c!),
        r2: Math.max(t.r!, b.r!),
        c2: Math.max(t.c!, b.c!),
      };
    }
    if (t.t === "id") {
      if (is("(")) {
        i++;
        const args: FNode[] = [];
        if (!is(")")) {
          args.push(expr());
          while (is(",")) {
            i++;
            args.push(expr());
          }
        }
        eat(")");
        return { k: "fn", name: t.v as string, args };
      }
      if (t.v === "TRUE" || t.v === "FALSE")
        return { k: "lit", v: t.v === "TRUE" };
      return { k: "lit", v: { err: "#NAME?" } };
    }
    if (t.t === "(") {
      const a = expr();
      eat(")");
      return a;
    }
    throw { err: "#NAME?" };
  };
  const postfix = () => {
    let a = primary();
    while (is("%")) {
      i++;
      a = { k: "bin", op: "/", a, b: { k: "lit", v: 100 } };
    }
    return a;
  };
  const unary = (): FNode => {
    if (is("-") || is("+")) {
      const op = toks[i++].t;
      return { k: "un", op, a: unary() };
    }
    return postfix();
  };
  const pow = level(unary, ["^"]);
  const mul = level(pow, ["*", "/"]);
  const add = level(mul, ["+", "-"]);
  const concat = level(add, ["&"]);
  const expr: () => FNode = level(concat, CMP);
  const node = expr();
  if (i < toks.length) throw { err: "#NAME?" };
  return node;
}

/**
 * Works out every cell of a sheet, once each: plain values, numbers with units, and formulas with SUM, AVERAGE, MIN,
 * MAX, COUNT, COUNTA, ROUND, ABS and IF. A cell that refers back to itself is #REF!
 */
function makeEval(get: (r: number, c: number) => string) {
  const memo = new Map<number, Val>();
  const busy = new Set<number>();
  const cell = (r: number, c: number): Val => {
    if (r < 0 || c < 0 || c > 16383) return { err: "#REF!" };
    const key = r * 16384 + c;
    const hit = memo.get(key);
    if (hit !== undefined) return hit;
    if (busy.has(key)) return { err: "#REF!" };
    const raw = get(r, c);
    let v: Val;
    if (raw.startsWith("=") && raw.length > 1) {
      busy.add(key);
      v = formula(raw.slice(1));
      busy.delete(key);
    } else if (!raw.trim()) v = null;
    else v = leadNum(raw) ?? raw;
    memo.set(key, v);
    return v;
  };
  const scalar = (v: Val | Val[]): Val => {
    if (Array.isArray(v)) throw { err: "#VALUE!" };
    return v;
  };
  const num = (v: Val | Val[]): number => {
    const s = scalar(v);
    if (s === null) return 0;
    if (typeof s === "number") return s;
    if (typeof s === "boolean") return s ? 1 : 0;
    if (isErr(s)) throw s;
    const n = leadNum(s);
    if (n === null) throw { err: "#VALUE!" };
    return n;
  };
  const text = (v: Val | Val[]) => {
    const s = scalar(v);
    if (isErr(s)) throw s;
    return s === null
      ? ""
      : typeof s === "boolean"
        ? s
          ? "TRUE"
          : "FALSE"
        : String(s);
  };
  const rank = (v: Val) =>
    typeof v === "number" ? 1 : typeof v === "string" ? 2 : 3;
  const compare = (x: Val, y: Val) => {
    if (isErr(x)) throw x;
    if (isErr(y)) throw y;
    const blank = (o: Val) =>
      typeof o === "string" ? "" : typeof o === "boolean" ? false : 0;
    const a = x === null ? blank(y) : x;
    const b = y === null ? blank(x) : y;
    if (rank(a) !== rank(b)) return rank(a) - rank(b);
    if (typeof a === "string")
      return a.toLowerCase().localeCompare(String(b).toLowerCase());
    return Number(a) - Number(b);
  };
  /** The numbers among a function's arguments – text and empty cells in a range are skipped, as in Excel */
  const nums = (args: FNode[]) => {
    const out: number[] = [];
    for (const a of args) {
      const v = ev(a);
      if (Array.isArray(v)) {
        for (const x of v) {
          if (isErr(x)) throw x;
          if (typeof x === "number") out.push(x);
        }
      } else if (a.k === "ref") {
        if (typeof v === "number") out.push(v);
      } else out.push(num(v));
    }
    return out;
  };
  const count = (args: FNode[], any: boolean) => {
    let n = 0;
    for (const a of args) {
      let v: Val | Val[];
      try {
        v = ev(a);
      } catch {
        continue;
      }
      for (const x of Array.isArray(v) ? v : [v])
        if (any ? x !== null && x !== "" : typeof x === "number") n++;
    }
    return n;
  };
  const call = (name: string, args: FNode[]): Val => {
    const sum = (l: number[]) => l.reduce((s, x) => s + x, 0);
    switch (name) {
      case "SUM":
        return sum(nums(args));
      case "AVERAGE": {
        const l = nums(args);
        if (!l.length) throw { err: "#DIV/0!" };
        return sum(l) / l.length;
      }
      case "MIN":
      case "MAX": {
        const l = nums(args);
        return l.length
          ? name === "MIN"
            ? Math.min(...l)
            : Math.max(...l)
          : 0;
      }
      case "COUNT":
        return count(args, false);
      case "COUNTA":
        return count(args, true);
      case "ABS":
        if (args.length !== 1) throw { err: "#VALUE!" };
        return Math.abs(num(ev(args[0])));
      case "ROUND": {
        if (args.length < 1 || args.length > 2) throw { err: "#VALUE!" };
        const x = num(ev(args[0]));
        const f = 10 ** Math.trunc(args[1] ? num(ev(args[1])) : 0);
        return (
          (Math.sign(x) * Math.round(Math.abs(x) * f * (1 + Number.EPSILON))) /
          f
        );
      }
      case "IF": {
        if (args.length < 2 || args.length > 3) throw { err: "#VALUE!" };
        const c = scalar(ev(args[0]));
        if (isErr(c)) throw c;
        if (typeof c === "string") throw { err: "#VALUE!" };
        const yes = typeof c === "boolean" ? c : !!c;
        const pick = yes ? args[1] : args[2];
        if (!pick) return yes ? 0 : false;
        const v = scalar(ev(pick));
        return v === null ? 0 : v;
      }
      default:
        throw { err: "#NAME?" };
    }
  };
  const ev = (n: FNode): Val | Val[] => {
    switch (n.k) {
      case "lit":
        if (isErr(n.v)) throw n.v;
        return n.v;
      case "ref": {
        const v = cell(n.r, n.c);
        if (isErr(v)) throw v;
        return v;
      }
      case "range": {
        if (n.r1 < 0 || n.c1 < 0) throw { err: "#REF!" };
        const out: Val[] = [];
        for (let r = n.r1; r <= n.r2; r++)
          for (let c = n.c1; c <= n.c2; c++) out.push(cell(r, c));
        return out;
      }
      case "un": {
        const x = num(ev(n.a));
        return n.op === "-" ? -x : x;
      }
      case "fn":
        return call(n.name, n.args);
      case "bin": {
        if (n.op === "&") return text(ev(n.a)) + text(ev(n.b));
        if (CMP.includes(n.op)) {
          const d = compare(scalar(ev(n.a)), scalar(ev(n.b)));
          return {
            "=": d === 0,
            "<>": d !== 0,
            "<": d < 0,
            ">": d > 0,
            "<=": d <= 0,
            ">=": d >= 0,
          }[n.op as "="]!;
        }
        const a = num(ev(n.a));
        const b = num(ev(n.b));
        if (n.op === "+") return a + b;
        if (n.op === "-") return a - b;
        if (n.op === "*") return a * b;
        if (n.op === "^") return a ** b;
        if (b === 0) throw { err: "#DIV/0!" };
        return a / b;
      }
    }
  };
  const formula = (src: string): Val => {
    try {
      const v = scalar(ev(parseFormula(src)));
      if (typeof v === "number" && !Number.isFinite(v)) return { err: "#NUM!" };
      return v === null ? 0 : v;
    } catch (e) {
      return isErr(e) ? { err: e.err } : { err: "#NAME?" };
    }
  };
  return cell;
}

const NUM_FMT: Record<
  ShellNumberFormat,
  { label: string; icon: IconSvgElement; f: Intl.NumberFormat }
> = {
  general: {
    label: "General",
    icon: TextFontIcon,
    f: new Intl.NumberFormat("en", { maximumFractionDigits: 10 }),
  },
  number: {
    label: "Number",
    icon: TextNumberSignIcon,
    f: new Intl.NumberFormat("en", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }),
  },
  percent: {
    label: "Percent",
    icon: PercentIcon,
    f: new Intl.NumberFormat("en", {
      style: "percent",
      maximumFractionDigits: 2,
    }),
  },
  currency: {
    label: "Currency",
    icon: Dollar01Icon,
    f: new Intl.NumberFormat("en", { style: "currency", currency: "USD" }),
  },
};
const statFmt = new Intl.NumberFormat("en", { maximumFractionDigits: 4 });
/** What a cell shows: a formula's result in the cell's number format; a typed value as typed, unless it's a plain number with a format */
const shown = (raw: string, v: Val, f: ShellNumberFormat = "general") => {
  if (isErr(v)) return v.err;
  if (typeof v === "boolean") return v ? "TRUE" : "FALSE";
  if (v === null) return "";
  if (typeof v === "number")
    return !raw.startsWith("=") && (f === "general" || !plainNum(raw))
      ? raw
      : NUM_FMT[f].f.format(v);
  return v;
};

const blankRow = (n: number) => Array.from({ length: n }, () => "");
/** The sheet a table opens as – rows as long as the longest one, the styles laid out by cell */
const toModel = (
  table: string[][],
  meta?: ShellArtifact["sheet"],
): SheetModel => {
  const cols = Math.max(1, ...table.map((r) => r.length));
  const data = table.map((r) =>
    Array.from({ length: cols }, (_, j) => r[j] ?? ""),
  );
  return {
    data,
    style: data.map((r, i) => r.map((_, j) => meta?.styles?.[`${i}:${j}`])),
    widths: meta?.widths ?? [],
  };
};
/** Back to a table – trailing empty rows and columns trimmed – and the widths and styles beside it */
const fromModel = (s: SheetModel) => {
  let R = 0;
  let C = 0;
  s.data.forEach((row, i) =>
    row.forEach((v, j) => {
      if (v !== "" || hasStyle(s.style[i]?.[j])) {
        R = Math.max(R, i + 1);
        C = Math.max(C, j + 1);
      }
    }),
  );
  const styles: Record<string, ShellCellStyle> = {};
  s.style.forEach((row, i) =>
    row?.forEach((st, j) => hasStyle(st) && (styles[`${i}:${j}`] = st!)),
  );
  return {
    table: R ? s.data.slice(0, R).map((row) => row.slice(0, C)) : [[""]],
    sheet: { widths: s.widths, styles },
  };
};
/** Writes values and / or styles; the sheet grows to take them */
const setCells = (
  s: SheetModel,
  cells: { r: number; c: number; v?: string; s?: ShellCellStyle }[],
): SheetModel => {
  const cols = Math.max(s.data[0]?.length ?? 0, ...cells.map((x) => x.c + 1));
  const rows = Math.max(s.data.length, ...cells.map((x) => x.r + 1));
  const data = Array.from({ length: rows }, (_, i) => {
    const row = s.data[i] ?? [];
    return row.length < cols ? [...row, ...blankRow(cols - row.length)] : row;
  });
  const style = Array.from({ length: rows }, (_, i) => s.style[i] ?? []);
  const fresh = new Set<number>();
  for (const x of cells) {
    if (!fresh.has(x.r)) {
      data[x.r] = [...data[x.r]];
      style[x.r] = [...style[x.r]];
      fresh.add(x.r);
    }
    if (x.v !== undefined) data[x.r][x.c] = x.v;
    if (x.s !== undefined) style[x.r][x.c] = x.s;
  }
  return { ...s, data, style };
};

type RefMap = (
  r: number,
  c: number,
  part: "one" | "start" | "end",
  fixedR: boolean,
  fixedC: boolean,
) => [number, number] | null;
const REFS =
  /(?<![A-Za-z0-9_$.])(\$?)([A-Za-z]{1,3})(\$?)(\d+)(?::(\$?)([A-Za-z]{1,3})(\$?)(\d+))?(?![A-Za-z0-9_(])/g;
/** Rewrites the refs of a formula – after rows or columns moved, or when it's copied elsewhere. A ref that's gone is #REF! */
const rewrite = (f: string, map: RefMap) =>
  f
    .split(/("(?:[^"]|"")*")/)
    .map((part, i) =>
      i % 2
        ? part
        : part.replace(
            REFS,
            (
              _m,
              d1: string,
              c1: string,
              d2: string,
              r1: string,
              d3?: string,
              c2?: string,
              d4?: string,
              r2?: string,
            ) => {
              const a = map(
                +r1 - 1,
                colIndex(c1),
                c2 === undefined ? "one" : "start",
                !!d2,
                !!d1,
              );
              if (c2 === undefined)
                return a ? `${d1}${colName(a[1])}${d2}${a[0] + 1}` : "#REF!";
              const b = map(+r2! - 1, colIndex(c2), "end", !!d4, !!d3);
              if (!a || !b || a[0] > b[0] || a[1] > b[1]) return "#REF!";
              return `${d1}${colName(a[1])}${d2}${a[0] + 1}:${d3}${colName(b[1])}${d4}${b[0] + 1}`;
            },
          ),
    )
    .join("");
const rewriteAll = (data: string[][], map: RefMap) =>
  data.map((row) =>
    row.some((v) => v.startsWith("="))
      ? row.map((v) => (v.startsWith("=") ? rewrite(v, map) : v))
      : row,
  );
/** Relative refs move with a copied formula, $-fixed ones stay */
const shiftFormula = (v: string, dr: number, dc: number) =>
  v.startsWith("=")
    ? rewrite(v, (r, c, _p, fr, fc) =>
        (fr ? r : r + dr) < 0 || (fc ? c : c + dc) < 0
          ? null
          : [fr ? r : r + dr, fc ? c : c + dc],
      )
    : v;

const insertAt = (
  s: SheetModel,
  axis: "row" | "col",
  at: number,
): SheetModel => {
  const cols = s.data[0]?.length ?? 1;
  const style = Array.from({ length: s.data.length }, (_, i) => [
    ...(s.style[i] ?? []),
  ]);
  if (axis === "row") {
    const data = [...s.data];
    while (data.length < at) (data.push(blankRow(cols)), style.push([]));
    data.splice(at, 0, blankRow(cols));
    style.splice(at, 0, []);
    return {
      ...s,
      data: rewriteAll(data, (r, c) => [r >= at ? r + 1 : r, c]),
      style,
    };
  }
  const data = s.data.map((row) => {
    const next =
      row.length < at ? [...row, ...blankRow(at - row.length)] : [...row];
    next.splice(at, 0, "");
    return next;
  });
  style.forEach((row) => row.length >= at && row.splice(at, 0, undefined));
  const widths = [...s.widths];
  if (widths.length >= at) widths.splice(at, 0, COL_W);
  return {
    data: rewriteAll(data, (r, c) => [r, c >= at ? c + 1 : c]),
    style,
    widths,
  };
};
const removeAt = (
  s: SheetModel,
  axis: "row" | "col",
  a: number,
  b: number,
): SheetModel => {
  const n = b - a + 1;
  const gone = (v: number, part: "one" | "start" | "end") =>
    v < a
      ? v
      : v > b
        ? v - n
        : part === "start"
          ? a
          : part === "end"
            ? a - 1
            : null;
  if (axis === "row") {
    const keep = (_: unknown, i: number) => i < a || i > b;
    const map: RefMap = (r, c, p) => {
      const v = gone(r, p);
      return v === null || v < 0 ? null : [v, c];
    };
    return {
      ...s,
      data: rewriteAll(s.data.filter(keep), map),
      style: Array.from(
        { length: s.data.length },
        (_, i) => s.style[i] ?? [],
      ).filter(keep),
    };
  }
  const keep = (_: unknown, j: number) => j < a || j > b;
  const map: RefMap = (r, c, p) => {
    const v = gone(c, p);
    return v === null || v < 0 ? null : [r, v];
  };
  const data = s.data.map((row) => row.filter(keep));
  return {
    data: rewriteAll(data[0]?.length ? data : data.map(() => [""]), map),
    style: s.style.map((row) => (row ?? []).filter(keep)),
    widths: s.widths.filter(keep),
  };
};
/**
 * Sorts the rows under the header by one column – numbers before text, empty cells last – rows keep their styles.
 * Like Excel, only the block of data around the header: it ends at the first empty row, so totals below a gap stay put
 */
const sortModel = (
  s: SheetModel,
  col: number,
  dir: 1 | -1,
  val: (r: number, c: number) => Val,
): SheetModel => {
  const gap = s.data.findIndex((row, i) => i > 0 && row.every((v) => v === ""));
  const n = gap < 0 ? s.data.length : gap;
  if (n < 3) return s;
  const style = Array.from(
    { length: s.data.length },
    (_, i) => s.style[i] ?? [],
  );
  const key = (r: number) => {
    const v = val(r, col);
    return v === null || v === "" ? null : v;
  };
  const rows = Array.from({ length: n - 1 }, (_, i) => i + 1).sort((a, b) => {
    const x = key(a);
    const y = key(b);
    if (x === null || y === null) return x === y ? a - b : x === null ? 1 : -1;
    if (typeof x === "number" && typeof y === "number")
      return (x - y) * dir || a - b;
    if ((typeof x === "number") !== (typeof y === "number"))
      return (typeof x === "number" ? -1 : 1) * dir;
    const tx = isErr(x) ? x.err : String(x);
    const ty = isErr(y) ? y.err : String(y);
    return (
      tx.localeCompare(ty, "en", { numeric: true, sensitivity: "base" }) *
        dir || a - b
    );
  });
  return {
    ...s,
    data: [s.data[0], ...rows.map((i) => s.data[i]), ...s.data.slice(n)],
    style: [style[0], ...rows.map((i) => style[i]), ...style.slice(n)],
  };
};

let measureCtx: CanvasRenderingContext2D | null = null;
/** A column wide enough for what's in it – what a double-click on its edge does */
const fitWidth = (
  texts: { text: string; bold: boolean }[],
  font: string,
  min = 40,
) => {
  measureCtx ??= document.createElement("canvas").getContext("2d");
  const w = (t: string, bold: boolean) => {
    if (!measureCtx) return t.length * 7;
    measureCtx.font = `${bold ? 500 : 400} 12px ${font}`;
    return measureCtx.measureText(t).width;
  };
  return Math.round(
    clamp(Math.max(0, ...texts.map((t) => w(t.text, t.bold))) + 20, min, 320),
  );
};

type Clip = {
  text: string;
  raw: string[][];
  style: (ShellCellStyle | undefined)[][];
  r: number;
  c: number;
  cut: boolean;
};

/**
 * The sheet as a small Excel: click, drag or Shift-click to select, the letters and numbers select whole columns and
 * rows; type to write, = for a formula, the formula bar above, Sum · Average · Count of the selection below. Copy, cut
 * and paste as tab-separated text, undo and redo, bold, a number format, sort, insert and delete, drag a column's edge
 */
function SheetView({
  name,
  table,
  meta,
  width,
  memo,
  onMemo,
  onChange,
}: {
  name: string;
  table: string[][];
  meta?: ShellArtifact["sheet"];
  width: number;
  memo?: SheetMemo;
  onMemo: (memo: SheetMemo) => void;
  onChange: (
    table: string[][],
    sheet: NonNullable<ShellArtifact["sheet"]>,
  ) => void;
}) {
  const uid = useId().replace(/:/g, "");
  const grid = useRef<HTMLTableElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const cellInput = useRef<HTMLInputElement>(null);
  const barInput = useRef<HTMLInputElement>(null);
  const [sheet, setSheet] = useState(() => toModel(table, meta));
  const hist = useRef<{ past: SheetModel[]; future: SheetModel[] }>({
    past: [],
    future: [],
  });
  const [sel, setSel] = useState<Sel>(
    memo?.sel ?? { r: 1, c: 0, r2: 1, c2: 0 },
  );
  const [edit, setEdit] = useState<{
    r: number;
    c: number;
    text: string;
    mode: "enter" | "edit";
    from: "cell" | "bar";
  } | null>(null);
  const editRef = useRef(edit);
  editRef.current = edit;
  const [nameBox, setNameBox] = useState<string | null>(null);
  const [fill, setFill] = useState<{ r: number; c: number } | null>(null);
  const fillRef = useRef(fill);
  fillRef.current = fill;
  const drag = useRef<null | "cells" | "cols" | "rows" | "fill">(null);
  const clip = useRef<Clip | null>(null);
  const clipDone = useRef(false);
  const [said, setSaid] = useState("");

  const R = sheet.data.length + 4;
  const C = (sheet.data[0]?.length ?? 1) + 2;
  const get = (r: number, c: number) => sheet.data[r]?.[c] ?? "";
  const styleOf = (r: number, c: number) => sheet.style[r]?.[c];
  const boldOf = (r: number, c: number) => styleOf(r, c)?.bold ?? r === 0;
  const value = useMemo(
    () => makeEval((r, c) => sheet.data[r]?.[c] ?? ""),
    [sheet.data],
  );
  const display = (r: number, c: number) =>
    shown(get(r, c), value(r, c), styleOf(r, c)?.format);
  const wOf = (c: number) => sheet.widths[c] ?? COL_W;
  const xOf = (c: number) => {
    let x = NUM_W;
    for (let i = 0; i < c; i++) x += wOf(i);
    return x;
  };
  const cellId = (r: number, c: number) => `${uid}-${r}-${c}`;
  const b = box(sel);
  const font = () =>
    grid.current
      ? getComputedStyle(grid.current).fontFamily
      : "Inter, system-ui, sans-serif";
  const fitCol = (s: SheetModel, c: number, min = 40) => {
    const ev = makeEval((r, j) => s.data[r]?.[j] ?? "");
    return fitWidth(
      s.data.map((row, r) => ({
        text: shown(row[c] ?? "", ev(r, c), s.style[r]?.[c]?.format),
        bold: s.style[r]?.[c]?.bold ?? r === 0,
      })),
      font(),
      min,
    );
  };

  // A sheet without widths of its own opens with each column fitted to what's in it
  useLayoutEffect(() => {
    if (meta?.widths?.length) return;
    setSheet((s) => ({
      ...s,
      widths: (s.data[0] ?? []).map((_, c) => Math.max(COL_W, fitCol(s, c))),
    }));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  // The selection is still there when you come back to the tab
  const selRef = useRef(sel);
  selRef.current = sel;
  const memoRef = useRef(onMemo);
  memoRef.current = onMemo;
  useEffect(() => () => memoRef.current({ sel: selRef.current }), []);

  const announce = (text: string) => {
    setSaid("");
    window.setTimeout(() => setSaid(text), 30);
  };
  const emit = (s: SheetModel) => {
    const out = fromModel(s);
    onChange(out.table, out.sheet);
  };
  const commit = (next: SheetModel, say?: string) => {
    hist.current.past.push(sheet);
    if (hist.current.past.length > 100) hist.current.past.shift();
    hist.current.future = [];
    setSheet(next);
    emit(next);
    if (say) announce(say);
  };
  const undo = () => {
    const prev = hist.current.past.pop();
    if (!prev) return announce("Nothing to undo");
    hist.current.future.push(sheet);
    setSheet(prev);
    emit(prev);
    announce("Undone");
  };
  const redo = () => {
    const next = hist.current.future.pop();
    if (!next) return announce("Nothing to redo");
    hist.current.past.push(sheet);
    setSheet(next);
    emit(next);
    announce("Redone");
  };

  /* Selection */
  const moveTo = (r: number, c: number) => {
    const nr = clamp(r, 0, R - 1);
    const nc = clamp(c, 0, C - 1);
    setSel({ r: nr, c: nc, r2: nr, c2: nc });
  };
  const extendTo = (r: number, c: number) =>
    setSel((s) => ({ ...s, r2: clamp(r, 0, R - 1), c2: clamp(c, 0, C - 1) }));
  /** ⌘ + an arrow: to the edge of the data, like Excel – the end of this block, or the start of the next one */
  const jump = (
    r: number,
    c: number,
    dr: number,
    dc: number,
  ): [number, number] => {
    const inside = (y: number, x: number) => y >= 0 && y < R && x >= 0 && x < C;
    const full = (y: number, x: number) => get(y, x) !== "";
    let y = r + dr;
    let x = c + dc;
    if (!inside(y, x)) return [r, c];
    if (full(r, c) && full(y, x)) {
      while (inside(y + dr, x + dc) && full(y + dr, x + dc))
        ((y += dr), (x += dc));
      return [y, x];
    }
    while (inside(y, x) && !full(y, x)) ((y += dr), (x += dc));
    return inside(y, x) ? [y, x] : [clamp(y, 0, R - 1), clamp(x, 0, C - 1)];
  };
  // The moving corner stays in sight – the sheet scrolls sideways inside itself, the panel up and down
  useEffect(() => {
    const w = wrap.current;
    if (!w) return;
    const left = xOf(sel.c2) - NUM_W;
    const right = xOf(sel.c2) + wOf(sel.c2);
    if (left < w.scrollLeft) w.scrollLeft = left;
    else if (right > w.scrollLeft + w.clientWidth)
      w.scrollLeft = right - w.clientWidth;
    if (grid.current?.contains(document.activeElement))
      revealY(document.getElementById(cellId(sel.r2, sel.c2)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sel]);

  /* Editing */
  const startEdit = (mode: "enter" | "edit", text?: string) =>
    setEdit({
      r: sel.r,
      c: sel.c,
      text: text ?? get(sel.r, sel.c),
      mode,
      from: "cell",
    });
  useLayoutEffect(() => {
    if (edit?.from !== "cell") return;
    const el = cellInput.current;
    if (!el || document.activeElement === el) return;
    el.focus({ preventScroll: true });
    el.setSelectionRange(el.value.length, el.value.length);
  }, [edit?.from, edit?.r, edit?.c]);
  const commitEdit = (dr = 0, dc = 0, refocus = true) => {
    const e = editRef.current;
    if (!e) return;
    editRef.current = null;
    setEdit(null);
    if (e.text !== get(e.r, e.c))
      commit(
        setCells(sheet, [{ r: e.r, c: e.c, v: e.text }]),
        e.text.startsWith("=")
          ? `${cellName(e.r, e.c)}: ${shown(e.text, makeEval((r, c) => (r === e.r && c === e.c ? e.text : get(r, c)))(e.r, e.c))}`
          : undefined,
      );
    if (dr || dc) moveTo(e.r + dr, e.c + dc);
    if (refocus) grid.current?.focus({ preventScroll: true });
  };
  const cancelEdit = () => {
    editRef.current = null;
    setEdit(null);
    grid.current?.focus({ preventScroll: true });
  };
  const onEditKey = (e: ReactKeyboardEvent<HTMLInputElement>) => {
    e.stopPropagation();
    const k = e.key;
    if (k === "Enter") {
      e.preventDefault();
      commitEdit(e.shiftKey ? -1 : 1, 0);
    } else if (k === "Tab") {
      e.preventDefault();
      commitEdit(0, e.shiftKey ? -1 : 1);
    } else if (k === "Escape") {
      e.preventDefault();
      cancelEdit();
    } else if (
      editRef.current?.mode === "enter" &&
      editRef.current.from === "cell" &&
      k.startsWith("Arrow")
    ) {
      // Typed into the cell: an arrow writes it and moves on, like Excel – after F2 it moves the caret
      e.preventDefault();
      commitEdit(
        k === "ArrowDown" ? 1 : k === "ArrowUp" ? -1 : 0,
        k === "ArrowRight" ? 1 : k === "ArrowLeft" ? -1 : 0,
      );
    }
  };

  /* Changes */
  const clearSel = () => {
    const cells = [];
    for (let r = b.r1; r <= Math.min(b.r2, sheet.data.length - 1); r++)
      for (let c = b.c1; c <= b.c2; c++)
        if (get(r, c) !== "") cells.push({ r, c, v: "" });
    if (cells.length) commit(setCells(sheet, cells), "Cleared");
  };
  const styleSel = (
    patch: (
      s: ShellCellStyle | undefined,
      r: number,
      c: number,
    ) => ShellCellStyle,
    say: string,
  ) => {
    const cells = [];
    for (let r = b.r1; r <= Math.min(b.r2, R - 5); r++)
      for (let c = b.c1; c <= Math.min(b.c2, C - 3); c++)
        cells.push({ r, c, s: patch(styleOf(r, c), r, c) });
    if (cells.length) commit(setCells(sheet, cells), say);
  };
  const toggleBold = () => {
    const on = !boldOf(sel.r, sel.c);
    styleSel((s) => ({ ...s, bold: on }), on ? "Bold" : "Not bold");
  };
  const setFormat = (f: ShellNumberFormat) =>
    styleSel((s) => ({ ...s, format: f }), `${NUM_FMT[f].label} format`);
  const sort = (dir: 1 | -1) => {
    commit(
      sortModel(sheet, sel.c, dir, value),
      `Sorted by column ${colName(sel.c)}, ${dir === 1 ? "A to Z" : "Z to A"}`,
    );
  };
  const insert = (axis: "row" | "col", after = false) => {
    const at =
      axis === "row" ? (after ? b.r2 + 1 : b.r1) : after ? b.c2 + 1 : b.c1;
    commit(
      insertAt(sheet, axis, at),
      axis === "row"
        ? `Row inserted ${after ? "below" : "above"} ${(after ? b.r2 : b.r1) + 1}`
        : `Column inserted ${after ? "right" : "left"} of ${colName(after ? b.c2 : b.c1)}`,
    );
    if (!after)
      setSel((x) =>
        axis === "row"
          ? { ...x, r: x.r + 1, r2: x.r2 + 1 }
          : { ...x, c: x.c + 1, c2: x.c2 + 1 },
      );
  };
  const remove = (axis: "row" | "col") => {
    const [a, z] = axis === "row" ? [b.r1, b.r2] : [b.c1, b.c2];
    const has =
      axis === "row" ? a < sheet.data.length : a < (sheet.data[0]?.length ?? 0);
    if (!has) return announce("Nothing to delete there");
    const next = removeAt(sheet, axis, a, z);
    commit(
      next,
      axis === "row"
        ? a === z
          ? `Row ${a + 1} deleted`
          : `Rows ${a + 1} to ${z + 1} deleted`
        : a === z
          ? `Column ${colName(a)} deleted`
          : `Columns ${colName(a)} to ${colName(z)} deleted`,
    );
    moveTo(axis === "row" ? a : sel.r, axis === "row" ? sel.c : a);
  };

  /* Clipboard – tab-separated text; within the sheet the formulas come along and their refs move */
  const copyText = (cut: boolean) => {
    const raw: string[][] = [];
    const style: (ShellCellStyle | undefined)[][] = [];
    const rows: string[] = [];
    for (let r = b.r1; r <= b.r2; r++) {
      raw.push([]);
      style.push([]);
      const out: string[] = [];
      for (let c = b.c1; c <= b.c2; c++) {
        raw[raw.length - 1].push(get(r, c));
        style[style.length - 1].push(styleOf(r, c));
        out.push(display(r, c));
      }
      rows.push(out.join("\t"));
    }
    const text = rows.join("\n");
    clip.current = { text, raw, style, r: b.r1, c: b.c1, cut };
    const n = (b.r2 - b.r1 + 1) * (b.c2 - b.c1 + 1);
    announce(
      `${cut ? "Cut" : "Copied"} ${n === 1 ? cellName(b.r1, b.c1) : `${number0.format(n)} cells`}`,
    );
    return text;
  };
  const pasteText = (text: string) => {
    const t = text.replace(/\r\n?/g, "\n").replace(/\n$/, "");
    const own = clip.current && clip.current.text === t ? clip.current : null;
    const rows = own ? own.raw : t.split("\n").map((l) => l.split("\t"));
    if (!rows.length || (!t && !own)) return;
    let next = sheet;
    if (own?.cut) {
      const gone = [];
      for (let i = 0; i < own.raw.length; i++)
        for (let j = 0; j < own.raw[i].length; j++)
          gone.push({
            r: own.r + i,
            c: own.c + j,
            v: "",
            s: undefined as ShellCellStyle | undefined,
          });
      next = setCells(next, gone);
    }
    const cells = [];
    for (let i = 0; i < rows.length; i++)
      for (let j = 0; j < rows[i].length; j++)
        cells.push({
          r: b.r1 + i,
          c: b.c1 + j,
          v:
            own && !own.cut
              ? shiftFormula(rows[i][j], b.r1 - own.r, b.c1 - own.c)
              : rows[i][j],
          ...(own ? { s: own.style[i]?.[j] } : {}),
        });
    const w = Math.max(...rows.map((r) => r.length));
    commit(
      setCells(next, cells),
      `Pasted ${cells.length === 1 ? "1 cell" : `${number0.format(cells.length)} cells`}`,
    );
    setSel({ r: b.r1, c: b.c1, r2: b.r1 + rows.length - 1, c2: b.c1 + w - 1 });
    if (own?.cut) clip.current = null;
  };

  /* Fill handle: drag the selection's corner to copy it down or across – formulas move their refs */
  const doFill = () => {
    const t = fillRef.current;
    setFill(null);
    if (!t) return;
    const down = Math.max(t.r - b.r2, b.r1 - t.r, 0);
    const across = Math.max(t.c - b.c2, b.c1 - t.c, 0);
    if (!down && !across) return;
    const h = b.r2 - b.r1 + 1;
    const w = b.c2 - b.c1 + 1;
    const to =
      down >= across
        ? {
            r1: Math.min(b.r1, t.r),
            r2: Math.max(b.r2, t.r),
            c1: b.c1,
            c2: b.c2,
          }
        : {
            r1: b.r1,
            r2: b.r2,
            c1: Math.min(b.c1, t.c),
            c2: Math.max(b.c2, t.c),
          };
    const cells = [];
    for (let r = to.r1; r <= to.r2; r++)
      for (let c = to.c1; c <= to.c2; c++) {
        if (r >= b.r1 && r <= b.r2 && c >= b.c1 && c <= b.c2) continue;
        const sr = b.r1 + ((((r - b.r1) % h) + h) % h);
        const sc = b.c1 + ((((c - b.c1) % w) + w) % w);
        cells.push({
          r,
          c,
          v: shiftFormula(get(sr, sc), r - sr, c - sc),
          s: styleOf(sr, sc),
        });
      }
    commit(
      setCells(sheet, cells),
      `Filled ${number0.format(cells.length)} cells`,
    );
    setSel({
      r: b.r1,
      c: b.c1,
      r2: to.r1 < b.r1 ? to.r1 : to.r2,
      c2: to.c1 < b.c1 ? to.c1 : to.c2,
    });
  };
  useEffect(() => {
    const up = () => {
      if (drag.current === "fill") doFill();
      drag.current = null;
    };
    window.addEventListener("pointerup", up);
    return () => window.removeEventListener("pointerup", up);
  });

  /* Column edges: drag to resize, double-click to fit */
  const resize = (c: number) => (e: ReactPointerEvent<HTMLSpanElement>) => {
    e.preventDefault();
    e.stopPropagation();
    const start = sheet;
    const x0 = e.clientX;
    const w0 = wOf(c);
    const el = e.currentTarget;
    // The pointer moves in screen pixels – in a scaled preview those are smaller or larger than the sheet's own
    const k =
      wrap.current && wrap.current.offsetWidth
        ? wrap.current.getBoundingClientRect().width /
            wrap.current.offsetWidth || 1
        : 1;
    el.setPointerCapture(e.pointerId);
    const widthsAt = (w: number) => {
      const widths = [...start.widths];
      while (widths.length <= c) widths.push(COL_W);
      widths[c] = Math.round(clamp(w, 32, 480));
      return widths;
    };
    const move = (ev: PointerEvent) =>
      setSheet((s) => ({ ...s, widths: widthsAt(w0 + (ev.clientX - x0) / k) }));
    const end = (ev: PointerEvent) => {
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerup", end);
      if (ev.clientX === x0) return setSheet(start);
      const next = { ...start, widths: widthsAt(w0 + (ev.clientX - x0) / k) };
      hist.current.past.push(start);
      hist.current.future = [];
      setSheet(next);
      emit(next);
      announce(`Column ${colName(c)} ${next.widths[c]} pixels wide`);
    };
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", end);
  };
  const autofit = (c: number) => {
    const widths = [...sheet.widths];
    while (widths.length <= c) widths.push(COL_W);
    widths[c] = fitCol(sheet, c);
    commit({ ...sheet, widths }, `Column ${colName(c)} fitted`);
  };

  /* Keys on the sheet */
  const onGridKey = (e: ReactKeyboardEvent<HTMLTableElement>) => {
    if (e.target !== grid.current) return;
    const mod = e.metaKey || e.ctrlKey;
    if (isMenuKey(e)) {
      e.preventDefault();
      return ctx.openOn(
        document.getElementById(cellId(sel.r, sel.c)),
        "cells",
        grid.current,
      );
    }
    const k = e.key;
    const arrow = (
      {
        ArrowUp: [-1, 0],
        ArrowDown: [1, 0],
        ArrowLeft: [0, -1],
        ArrowRight: [0, 1],
      } as Record<string, [number, number]>
    )[k];
    if (arrow) {
      e.preventDefault();
      const [dr, dc] = arrow;
      if (e.shiftKey) {
        const [r, c] = mod
          ? jump(sel.r2, sel.c2, dr, dc)
          : [sel.r2 + dr, sel.c2 + dc];
        return extendTo(r, c);
      }
      const [r, c] = mod
        ? jump(sel.r, sel.c, dr, dc)
        : [sel.r + dr, sel.c + dc];
      return moveTo(r, c);
    }
    if (k === "Tab") {
      // At the sheet's first or last column Tab moves on to the next control, so the sheet never traps the focus
      const c = sel.c + (e.shiftKey ? -1 : 1);
      if (c < 0 || c >= C) return;
      e.preventDefault();
      return moveTo(sel.r, c);
    }
    if (k === "Enter") {
      e.preventDefault();
      return moveTo(sel.r + (e.shiftKey ? -1 : 1), sel.c);
    }
    if (k === "F2") {
      e.preventDefault();
      return startEdit("edit");
    }
    if (k === "Delete" || k === "Backspace") {
      e.preventDefault();
      return clearSel();
    }
    if (k === "Home") {
      e.preventDefault();
      return moveTo(mod ? 0 : sel.r, 0);
    }
    if (k === "End" && mod) {
      e.preventDefault();
      return moveTo(sheet.data.length - 1, (sheet.data[0]?.length ?? 1) - 1);
    }
    if (mod && !e.altKey) {
      const key = k.toLowerCase();
      if (key === "a") {
        e.preventDefault();
        setSel({
          r: 0,
          c: 0,
          r2: sheet.data.length - 1,
          c2: (sheet.data[0]?.length ?? 1) - 1,
        });
        announce("All cells selected");
      } else if (key === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
      } else if (key === "y") {
        e.preventDefault();
        redo();
      } else if (key === "b") {
        e.preventDefault();
        toggleBold();
      } else if (key === "c" || key === "x") {
        // The copy event follows and fills the clipboard; where it doesn't, the Clipboard API does
        const text = copyText(key === "x");
        clipDone.current = false;
        window.setTimeout(
          () =>
            !clipDone.current &&
            navigator.clipboard?.writeText(text).catch(() => {}),
        );
      } else if (key === "v") {
        clipDone.current = false;
        window.setTimeout(async () => {
          if (clipDone.current) return;
          let text: string | undefined;
          try {
            text = await navigator.clipboard.readText();
          } catch {
            text = undefined;
          }
          pasteText(text ?? clip.current?.text ?? "");
        });
      }
      return;
    }
    if (k.length === 1 && !e.altKey) {
      e.preventDefault();
      startEdit("enter", k);
    }
  };

  /* Pointer: cells, letters, numbers */
  const pointCell = (r: number, c: number, e: ReactPointerEvent) => {
    const ed = editRef.current;
    // Writing a formula: a click puts that cell's name into it
    if (
      ed &&
      ed.text.startsWith("=") &&
      /[=(+\-*/,:<>&^]$/.test(ed.text) &&
      !(ed.r === r && ed.c === c)
    ) {
      e.preventDefault();
      setEdit({ ...ed, text: ed.text + cellName(r, c) });
      (ed.from === "bar" ? barInput : cellInput).current?.focus();
      return;
    }
    if (e.button !== 0) return;
    if (ed && (ed.r !== r || ed.c !== c)) commitEdit(0, 0, false);
    if (ed && ed.r === r && ed.c === c) return;
    if (e.shiftKey) extendTo(r, c);
    else {
      setSel({ r, c, r2: r, c2: c });
      drag.current = "cells";
    }
  };
  const pointCol = (c: number, e: ReactPointerEvent) => {
    if (e.button !== 0) return;
    commitEdit(0, 0, false);
    if (e.shiftKey) setSel((s) => ({ ...s, r: 0, r2: R - 1, c2: c }));
    else setSel({ r: 0, c, r2: R - 1, c2: c });
    drag.current = "cols";
  };
  const pointRow = (r: number, e: ReactPointerEvent) => {
    if (e.button !== 0) return;
    commitEdit(0, 0, false);
    if (e.shiftKey) setSel((s) => ({ ...s, c: 0, c2: C - 1, r2: r }));
    else setSel({ r, c: 0, r2: r, c2: C - 1 });
    drag.current = "rows";
  };

  // What the status bar says: only once more than one cell is selected
  const many = b.r1 !== b.r2 || b.c1 !== b.c2;
  const stats = useMemo(() => {
    if (!many) return null;
    const l: number[] = [];
    for (let r = b.r1; r <= Math.min(b.r2, sheet.data.length - 1); r++)
      for (let c = b.c1; c <= b.c2; c++) {
        const v = value(r, c);
        if (typeof v === "number") l.push(v);
      }
    if (!l.length) return null;
    const sum = l.reduce((s, x) => s + x, 0);
    return { sum, avg: sum / l.length, n: l.length };
  }, [many, b.r1, b.r2, b.c1, b.c2, value, sheet.data.length]);

  const refText = many
    ? `${cellName(b.r1, b.c1)}:${cellName(b.r2, b.c2)}`
    : cellName(sel.r, sel.c);
  const active = styleOf(sel.r, sel.c);
  const fmt = active?.format ?? "general";
  const total = xOf(C);
  const ring0 = {
    left: xOf(b.c1),
    top: HEAD_H + b.r1 * ROW_H,
    width: xOf(b.c2 + 1) - xOf(b.c1),
    height: (b.r2 - b.r1 + 1) * ROW_H,
  };
  // The rings sit exactly on the cells' edges: measured from the cells themselves once they're laid out, so rounding,
  // the half-pixel gridlines or a taller row can never shift them
  const [boxes, setBoxes] = useState<{
    range: CSSProperties;
    cell: CSSProperties;
  } | null>(null);
  useLayoutEffect(() => {
    const box = wrap.current;
    if (!box) return;
    const at = (r: number, c: number) =>
      document.getElementById(cellId(r, c))?.getBoundingClientRect();
    const a = at(b.r1, b.c1);
    const z = at(b.r2, b.c2);
    const one = at(sel.r, sel.c);
    if (!a || !z || !one) return setBoxes((was) => (was === null ? was : null));
    const o = box.getBoundingClientRect();
    // Measured on screen, drawn in the sheet's own pixels – in a scaled preview (zoom or transform) they differ by this
    const k = box.offsetWidth ? o.width / box.offsetWidth || 1 : 1;
    const x = (v: number) => (v - o.left) / k + box.scrollLeft;
    const y = (v: number) => (v - o.top) / k + box.scrollTop;
    const next = {
      range: {
        left: x(a.left),
        top: y(a.top),
        width: (z.right - a.left) / k,
        height: (z.bottom - a.top) / k,
      },
      cell: {
        left: x(one.left),
        top: y(one.top),
        width: one.width / k,
        height: one.height / k,
      },
    };
    // Only when something moved – this runs after every render (column widths, edits and rows all shift cells)
    setBoxes((was) =>
      was && JSON.stringify(was) === JSON.stringify(next) ? was : next,
    );
  });
  const ring = boxes?.range ?? ring0;
  const fillBox = fill
    ? (() => {
        const down =
          Math.max(fill.r - b.r2, b.r1 - fill.r, 0) >=
          Math.max(fill.c - b.c2, b.c1 - fill.c, 0);
        const r1 = down ? Math.min(b.r1, fill.r) : b.r1;
        const r2 = down ? Math.max(b.r2, fill.r) : b.r2;
        const c1 = down ? b.c1 : Math.min(b.c1, fill.c);
        const c2 = down ? b.c2 : Math.max(b.c2, fill.c);
        return {
          left: xOf(c1),
          top: HEAD_H + r1 * ROW_H,
          width: xOf(c2 + 1) - xOf(c1),
          height: (r2 - r1 + 1) * ROW_H,
        };
      })()
    : null;
  const L = colName(sel.c);
  const roomy = width >= 360;
  const rowsCols: MenuItem[] = [
    {
      label: "Insert row above",
      icon: InsertRowUpIcon,
      onSelect: () => insert("row"),
    },
    {
      label: "Insert column left",
      icon: InsertColumnLeftIcon,
      onSelect: () => insert("col"),
    },
    {
      label: b.r1 === b.r2 ? "Delete row" : "Delete rows",
      icon: RowDeleteIcon,
      divider: !roomy,
      onSelect: () => remove("row"),
    },
    {
      label: b.c1 === b.c2 ? "Delete column" : "Delete columns",
      icon: ColumnDeleteIcon,
      onSelect: () => remove("col"),
    },
  ];
  const back = () =>
    window.setTimeout(() => grid.current?.focus({ preventScroll: true }));

  /* Right-click: a cell, letter or number outside the selection gets selected first – like Excel */
  const ctx = useContextMenu<"cells">();
  const within = (r: number, c: number) =>
    r >= b.r1 && r <= b.r2 && c >= b.c1 && c <= b.c2;
  const sheetMenu = (): CtxItem[] => [
    {
      label: "Cut",
      icon: Scissor01Icon,
      keys: `${MOD}X`,
      onSelect: () => toClipboard(copyText(true)),
    },
    {
      label: "Copy",
      icon: Copy01Icon,
      keys: `${MOD}C`,
      onSelect: () => toClipboard(copyText(false)),
    },
    {
      label: "Paste",
      icon: ClipboardIcon,
      keys: `${MOD}V`,
      onSelect: async () => {
        const t = await fromClipboard();
        if (t === null && !clip.current)
          return announce(
            `Press ${MOD}V to paste – the browser keeps the clipboard to itself here`,
          );
        pasteText(t ?? clip.current?.text ?? "");
      },
    },
    // Grouped like Google Sheets: the clipboard on top, everything else one level down, so the menu stays short
    {
      label: "Insert",
      icon: PlusSignIcon,
      divider: true,
      submenu: [
        {
          label: "Row above",
          icon: InsertRowUpIcon,
          onSelect: () => insert("row"),
        },
        {
          label: "Row below",
          icon: InsertRowDownIcon,
          onSelect: () => insert("row", true),
        },
        {
          label: "Column left",
          icon: InsertColumnLeftIcon,
          divider: true,
          onSelect: () => insert("col"),
        },
        {
          label: "Column right",
          icon: InsertColumnRightIcon,
          onSelect: () => insert("col", true),
        },
      ],
    },
    {
      label: "Delete",
      icon: Delete02Icon,
      submenu: [
        {
          label: "Clear contents",
          icon: EraserIcon,
          keys: "⌫",
          onSelect: clearSel,
        },
        {
          label: b.r1 === b.r2 ? "Delete row" : "Delete rows",
          icon: RowDeleteIcon,
          divider: true,
          onSelect: () => remove("row"),
        },
        {
          label: b.c1 === b.c2 ? "Delete column" : "Delete columns",
          icon: ColumnDeleteIcon,
          onSelect: () => remove("col"),
        },
      ],
    },
    {
      label: "Sort",
      icon: SortingAZ01Icon,
      submenu: [
        {
          label: `A to Z by ${L}`,
          icon: SortingAZ01Icon,
          onSelect: () => sort(1),
        },
        {
          label: `Z to A by ${L}`,
          icon: SortingZA01Icon,
          onSelect: () => sort(-1),
        },
      ],
    },
    {
      label: "Format",
      icon: TextFontIcon,
      submenu: [
        {
          label: "Bold",
          icon: TextBoldIcon,
          keys: `${MOD}B`,
          checked: boldOf(sel.r, sel.c),
          onSelect: toggleBold,
        },
        ...(Object.keys(NUM_FMT) as ShellNumberFormat[]).map((f, i) => ({
          label: NUM_FMT[f].label,
          icon: NUM_FMT[f].icon,
          checked: f === fmt,
          divider: i === 0,
          onSelect: () => setFormat(f),
        })),
      ],
    },
  ];
  const withFocus = (items: MenuItem[]) =>
    items.map((i) => ({ ...i, onSelect: () => (i.onSelect?.(), back()) }));

  return (
    <div className="ui-asw-sh">
      <Toolbar label="Sheet" className="ui-asw-sh__tools">
        <span className="ui-asw-tb__group">
          <Tool
            icon={Undo02Icon}
            label="Undo"
            keys={`${MOD}Z`}
            disabled={!hist.current.past.length}
            onClick={undo}
          />
          <Tool
            icon={Redo02Icon}
            label="Redo"
            keys={`${SHIFT}${MOD}Z`}
            disabled={!hist.current.future.length}
            onClick={redo}
          />
        </span>
        <span className="ui-asw-tb__group">
          <Tool
            icon={TextBoldIcon}
            label="Bold"
            keys={`${MOD}B`}
            pressed={boldOf(sel.r, sel.c)}
            onClick={toggleBold}
          />
          <MenuTool
            icon={NUM_FMT[fmt].icon}
            label="Number format"
            text={width >= 400 ? NUM_FMT[fmt].label : undefined}
            items={withFocus(
              (Object.keys(NUM_FMT) as ShellNumberFormat[]).map((f) => ({
                label: NUM_FMT[f].label,
                icon: NUM_FMT[f].icon,
                checked: f === fmt,
                onSelect: () => setFormat(f),
              })),
            )}
          />
        </span>
        <span className="ui-asw-tb__group">
          <Tool
            icon={SortingAZ01Icon}
            label={`Sort A to Z by ${L}`}
            onClick={() => sort(1)}
          />
          <Tool
            icon={SortingZA01Icon}
            label={`Sort Z to A by ${L}`}
            onClick={() => sort(-1)}
          />
        </span>
        <span className="ui-asw-tb__group">
          {roomy ? (
            <>
              <Tool
                icon={InsertRowUpIcon}
                label="Insert row above"
                onClick={() => insert("row")}
              />
              <Tool
                icon={InsertColumnLeftIcon}
                label="Insert column left"
                onClick={() => insert("col")}
              />
              <MenuTool
                icon={Delete02Icon}
                label="Delete"
                items={withFocus(rowsCols.slice(2))}
                align="end"
              />
            </>
          ) : (
            <MenuTool
              icon={GridTableIcon}
              label="Rows and columns"
              items={withFocus(rowsCols)}
              align="end"
            />
          )}
        </span>
      </Toolbar>

      {/* The name box and the formula bar */}
      <div className="ui-asw-sh__bar">
        <input
          className="ui-asw-sh__ref ui-asw-num"
          aria-label="Cell reference – type one to go there"
          value={nameBox ?? refText}
          spellCheck={false}
          onFocus={(e) => {
            setNameBox(refText);
            e.currentTarget.select();
          }}
          onChange={(e) => setNameBox(e.target.value)}
          onBlur={() => setNameBox(null)}
          onKeyDown={(e) => {
            e.stopPropagation();
            if (e.key === "Escape") {
              e.preventDefault();
              grid.current?.focus();
            }
            if (e.key !== "Enter") return;
            e.preventDefault();
            const m =
              /^\s*([A-Za-z]{1,3})(\d+)(?::([A-Za-z]{1,3})(\d+))?\s*$/.exec(
                nameBox ?? "",
              );
            if (!m) return announce("Not a cell reference");
            const r = +m[2] - 1;
            const c = colIndex(m[1]);
            setSel({
              r: clamp(r, 0, R - 1),
              c: clamp(c, 0, C - 1),
              r2: clamp(m[4] ? +m[4] - 1 : r, 0, R - 1),
              c2: clamp(m[3] ? colIndex(m[3]) : c, 0, C - 1),
            });
            grid.current?.focus();
          }}
        />
        <span className="ui-asw-sh__fx" aria-hidden="true">
          fx
        </span>
        <input
          ref={barInput}
          className="ui-asw-sh__formula"
          aria-label={`Contents of ${cellName(sel.r, sel.c)}`}
          value={edit ? edit.text : get(sel.r, sel.c)}
          spellCheck={false}
          onFocus={() =>
            !editRef.current &&
            setEdit({
              r: sel.r,
              c: sel.c,
              text: get(sel.r, sel.c),
              mode: "edit",
              from: "bar",
            })
          }
          onChange={(e) =>
            setEdit((x) => ({
              ...(x ?? { r: sel.r, c: sel.c, mode: "edit" as const }),
              from: "bar",
              text: e.target.value,
            }))
          }
          onKeyDown={onEditKey}
          onBlur={() => commitEdit(0, 0, false)}
        />
      </div>

      <div ref={wrap} className="ui-asw-sh__wrap">
        <table
          ref={grid}
          className="ui-asw-sh__grid"
          role="grid"
          tabIndex={0}
          aria-label={`${name}, sheet`}
          aria-multiselectable="true"
          aria-rowcount={R + 1}
          aria-colcount={C + 1}
          aria-activedescendant={cellId(sel.r, sel.c)}
          style={{ width: `max(100%, ${total}px)` }}
          onKeyDown={onGridKey}
          onContextMenu={(e) => {
            // The field you type in keeps the browser's own menu – spelling, emoji
            if ((e.target as HTMLElement).closest("input")) return;
            e.preventDefault();
            commitEdit(0, 0, false);
            grid.current?.focus({ preventScroll: true });
            ctx.open(e.clientX, e.clientY, "cells", grid.current);
          }}
          onCopy={(e) => {
            if (e.target !== grid.current) return;
            e.preventDefault();
            clipDone.current = true;
            try {
              e.clipboardData.setData(
                "text/plain",
                clip.current?.text ?? copyText(false),
              );
            } catch {
              /* the internal clipboard still has it */
            }
          }}
          onCut={(e) => {
            if (e.target !== grid.current) return;
            e.preventDefault();
            clipDone.current = true;
            try {
              e.clipboardData.setData(
                "text/plain",
                clip.current?.text ?? copyText(true),
              );
            } catch {
              /* the internal clipboard still has it */
            }
          }}
          onPaste={(e) => {
            if (e.target !== grid.current) return;
            e.preventDefault();
            clipDone.current = true;
            let text = "";
            try {
              text = e.clipboardData.getData("text/plain");
            } catch {
              text = clip.current?.text ?? "";
            }
            pasteText(text || clip.current?.text || "");
          }}
        >
          <colgroup>
            <col style={{ width: NUM_W }} />
            {Array.from({ length: C }, (_, c) => (
              <col key={c} style={{ width: wOf(c) }} />
            ))}
            <col />
          </colgroup>
          <thead>
            <tr role="row" aria-rowindex={1}>
              <th
                className="ui-asw-sh__corner"
                role="columnheader"
                aria-label="Select all"
                onPointerDown={(e) => {
                  if (e.button !== 0) return;
                  commitEdit(0, 0, false);
                  setSel({ r: 0, c: 0, r2: R - 1, c2: C - 1 });
                }}
              />
              {Array.from({ length: C }, (_, c) => (
                <th
                  key={c}
                  className="ui-asw-sh__col"
                  role="columnheader"
                  aria-label={`Column ${colName(c)}`}
                  data-on={(c >= b.c1 && c <= b.c2) || undefined}
                  onPointerDown={(e) => pointCol(c, e)}
                  onContextMenu={() =>
                    !within(0, c) || b.r1 !== 0 || b.r2 !== R - 1
                      ? setSel({ r: 0, c, r2: R - 1, c2: c })
                      : undefined
                  }
                  onPointerEnter={() =>
                    drag.current === "cols" && setSel((s) => ({ ...s, c2: c }))
                  }
                >
                  {colName(c)}
                  <span
                    className="ui-asw-sh__resize"
                    aria-hidden="true"
                    onPointerDown={resize(c)}
                    onDoubleClick={() => autofit(c)}
                  />
                </th>
              ))}
              <th className="ui-asw-sh__pad" aria-hidden="true" />
            </tr>
          </thead>
          <tbody>
            {Array.from({ length: R }, (_, r) => (
              <tr key={r} role="row" aria-rowindex={r + 2}>
                <th
                  className="ui-asw-sh__row ui-asw-num"
                  role="rowheader"
                  aria-label={`Row ${r + 1}`}
                  data-on={(r >= b.r1 && r <= b.r2) || undefined}
                  onPointerDown={(e) => pointRow(r, e)}
                  onContextMenu={() =>
                    !within(r, 0) || b.c1 !== 0 || b.c2 !== C - 1
                      ? setSel({ r, c: 0, r2: r, c2: C - 1 })
                      : undefined
                  }
                  onPointerEnter={() =>
                    drag.current === "rows" && setSel((s) => ({ ...s, r2: r }))
                  }
                >
                  {r + 1}
                </th>
                {Array.from({ length: C }, (_, c) => {
                  const v = value(r, c);
                  const text = shown(get(r, c), v, styleOf(r, c)?.format);
                  const inSel =
                    r >= b.r1 && r <= b.r2 && c >= b.c1 && c <= b.c2;
                  const editing = edit && edit.r === r && edit.c === c;
                  return (
                    <td
                      key={c}
                      id={cellId(r, c)}
                      role="gridcell"
                      className="ui-asw-sh__cell"
                      aria-selected={inSel}
                      aria-label={`${text || "Empty"}, ${cellName(r, c)}`}
                      data-num={typeof v === "number" || undefined}
                      data-err={isErr(v) || undefined}
                      data-bold={boldOf(r, c) || undefined}
                      data-sel={
                        (inSel && many && !(r === sel.r && c === sel.c)) ||
                        undefined
                      }
                      onPointerDown={(e) => pointCell(r, c, e)}
                      onContextMenu={() =>
                        !within(r, c) && setSel({ r, c, r2: r, c2: c })
                      }
                      onPointerEnter={() => {
                        if (drag.current === "cells") extendTo(r, c);
                        else if (drag.current === "fill") setFill({ r, c });
                      }}
                      onDoubleClick={() => {
                        setSel({ r, c, r2: r, c2: c });
                        setEdit({
                          r,
                          c,
                          text: get(r, c),
                          mode: "edit",
                          from: "cell",
                        });
                      }}
                    >
                      {editing && edit.from === "cell" ? (
                        <input
                          ref={cellInput}
                          className="ui-asw-sh__input"
                          aria-label={`Edit ${cellName(r, c)}`}
                          value={edit.text}
                          spellCheck={false}
                          style={{ minWidth: wOf(c) }}
                          size={Math.max(1, edit.text.length + 2)}
                          onChange={(e) =>
                            setEdit((x) => x && { ...x, text: e.target.value })
                          }
                          onKeyDown={onEditKey}
                          onBlur={(e) => {
                            if (e.relatedTarget === barInput.current)
                              setEdit((x) => x && { ...x, from: "bar" });
                            else commitEdit(0, 0, false);
                          }}
                        />
                      ) : editing ? (
                        edit.text
                      ) : (
                        text
                      )}
                    </td>
                  );
                })}
                <td className="ui-asw-sh__pad" aria-hidden="true" />
              </tr>
            ))}
          </tbody>
        </table>
        {/* Like Google Sheets: the first cell gets the strong ring, the rest of the range a thin one */}
        {many && (
          <div
            className="ui-asw-sh__ring ui-asw-sh__ring--cell"
            style={boxes?.cell}
            aria-hidden="true"
          />
        )}
        <div
          className="ui-asw-sh__ring"
          data-range={many || undefined}
          style={ring}
          aria-hidden="true"
        >
          <span
            className="ui-asw-sh__handle"
            onPointerDown={(e) => {
              if (e.button !== 0) return;
              e.preventDefault();
              e.stopPropagation();
              commitEdit(0, 0, false);
              drag.current = "fill";
              grid.current?.focus({ preventScroll: true });
            }}
          />
        </div>
        {fillBox && (
          <div className="ui-asw-sh__fill" style={fillBox} aria-hidden="true" />
        )}
      </div>

      {ctx.render(() => ({ label: `${refText}`, items: sheetMenu() }))}
      <div
        className="ui-asw-sh__status ui-asw-num"
        aria-hidden={!stats || undefined}
      >
        {stats && (
          <>
            <span>
              Sum <b>{statFmt.format(stats.sum)}</b>
            </span>
            <span>
              Average <b>{statFmt.format(stats.avg)}</b>
            </span>
            <span>
              Count <b>{number0.format(stats.n)}</b>
            </span>
          </>
        )}
      </div>
      <span className="ui-asw-sr" role="status" aria-live="polite">
        {said}
      </span>
    </div>
  );
}

/* ---------- Folder: a small Finder ---------- */

type Entry = Exclude<ShellFolderEntry, string>;
type SortKey = "name" | "kind" | "size" | "modified";
type FolderMemo = {
  path: string[];
  view: "list" | "icons";
  sort: { key: SortKey; dir: 1 | -1 };
};
const entryOf = (e: ShellFolderEntry): Entry =>
  typeof e === "string" ? { name: e } : e;
const extOf = (n: string) =>
  n.includes(".") ? n.slice(n.lastIndexOf(".") + 1).toLowerCase() : "";
const KINDS: Record<string, string> = {
  md: "Markdown",
  csv: "CSV",
  tsv: "TSV",
  xlsx: "Spreadsheet",
  json: "JSON",
  yaml: "YAML",
  yml: "YAML",
  env: "Environment",
  png: "PNG image",
  jpg: "JPEG image",
  svg: "SVG image",
  pdf: "PDF",
  ts: "TypeScript",
  js: "JavaScript",
  txt: "Plain text",
};
const isFolder = (e: Entry) =>
  e.kind === "Folder" || !!e.files || !extOf(e.name);
const kindOf = (e: Entry) =>
  e.kind ??
  (isFolder(e)
    ? "Folder"
    : (KINDS[extOf(e.name)] ?? `${extOf(e.name).toUpperCase()} file`));
const bytesOf = (s: Entry["size"]) =>
  typeof s === "number"
    ? s
    : typeof s === "string"
      ? (leadNum(s) ?? 0) *
        ({ k: 1e3, m: 1e6, g: 1e9 }[
          /([kmg])b/i.exec(s)?.[1].toLowerCase() ?? ""
        ] ?? 1)
      : -1;
const sizeText = (e: Entry, inside?: number) => {
  if (isFolder(e))
    return inside === undefined
      ? "--"
      : inside === 1
        ? "1 item"
        : `${number0.format(inside)} items`;
  if (typeof e.size === "string") return e.size;
  if (typeof e.size !== "number") return "--";
  if (e.size < 1000) return `${e.size} bytes`;
  const [n, u] =
    e.size < 1e6
      ? [e.size / 1e3, "KB"]
      : e.size < 1e9
        ? [e.size / 1e6, "MB"]
        : [e.size / 1e9, "GB"];
  return `${n < 10 ? n.toFixed(1).replace(/\.0$/, "") : Math.round(n)} ${u}`;
};
const timeFmt = new Intl.DateTimeFormat("en", {
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});
const dayFmt = new Intl.DateTimeFormat("en", {
  month: "short",
  day: "numeric",
});
const yearFmt = new Intl.DateTimeFormat("en", {
  month: "short",
  day: "numeric",
  year: "numeric",
});
/** Today, 09:41 · Yesterday, 17:02 · Nov 3 – as Finder says it, a bit shorter */
const whenText = (m: Entry["modified"], now: number, short = false) => {
  if (m === undefined) return "--";
  const d = new Date(m);
  if (Number.isNaN(+d)) return String(m);
  const day = (t: Date) =>
    new Date(t.getFullYear(), t.getMonth(), t.getDate()).getTime();
  const days = Math.round((day(new Date(now)) - day(d)) / 86_400_000);
  if (days === 0)
    return short ? timeFmt.format(d) : `Today, ${timeFmt.format(d)}`;
  if (days === 1)
    return short ? "Yesterday" : `Yesterday, ${timeFmt.format(d)}`;
  return d.getFullYear() === new Date(now).getFullYear()
    ? dayFmt.format(d)
    : yearFmt.format(d);
};
const SORTS: { key: SortKey; label: string; icon: IconSvgElement }[] = [
  { key: "name", label: "Name", icon: TextFontIcon },
  { key: "kind", label: "Kind", icon: File01Icon },
  { key: "size", label: "Size", icon: HardDriveIcon },
  { key: "modified", label: "Date modified", icon: Calendar03Icon },
];

/**
 * A folder like Finder's: back and forward through its subfolders, the path, a list or icons, a search that filters as
 * you type and a sort. Click, ⌘-click and Shift-click to select; Enter or a double-click opens a file in its own tab
 * and a folder in place; Space shows a Quick Look of the selected file
 */
function FolderView({
  name,
  entries,
  lookup,
  width,
  memo,
  onMemo,
  onOpen,
  onSeed,
  onChange,
}: {
  name: string;
  entries: ShellFolderEntry[];
  lookup: (name: string) => ShellArtifact | undefined;
  width: number;
  memo?: FolderMemo;
  onMemo: (memo: FolderMemo) => void;
  onOpen: (name: string) => void;
  onSeed: (name: string, artifact: ShellArtifact) => void;
  /** The folder's entries after a rename, a duplicate, a new folder or a move to the trash */
  onChange: (files: ShellFolderEntry[]) => void;
}) {
  const uid = useId().replace(/:/g, "");
  const listRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const [path, setPath] = useState<string[]>(memo?.path ?? []);
  const [hist, setHist] = useState<{ back: string[][]; fwd: string[][] }>({
    back: [],
    fwd: [],
  });
  const [view, setView] = useState<"list" | "icons">(memo?.view ?? "list");
  const [sort, setSort] = useState<FolderMemo["sort"]>(
    memo?.sort ?? { key: "name", dir: 1 },
  );
  const [q, setQ] = useState("");
  const [sel, setSel] = useState<string[]>([]);
  const [cur, setCur] = useState<string | null>(null);
  const anchor = useRef<string | null>(null);
  const [look, setLook] = useState(false);
  const [renaming, setRenaming] = useState<string | null>(null);
  const [said, setSaid] = useState("");
  const typed = useRef({ text: "", at: 0 });
  const now = Date.now();

  const state = useRef({ path, view, sort });
  state.current = { path, view, sort };
  const memoRef = useRef(onMemo);
  memoRef.current = onMemo;
  useEffect(() => () => memoRef.current(state.current), []);

  /** The entries of a subfolder: its own, or those of the artifact by its name */
  const inside = (e: Entry) => (e.files ?? lookup(e.name)?.files)?.map(entryOf);
  const listAt = (p: string[]) => {
    let list = entries.map(entryOf);
    for (const seg of p) {
      const e = list.find((x) => x.name === seg);
      const sub = e && inside(e);
      if (!sub) return null;
      list = sub;
    }
    return list;
  };
  const here = listAt(path) ?? entries.map(entryOf);
  const items = (() => {
    const needle = q.trim().toLowerCase();
    const l = needle
      ? here.filter((e) => e.name.toLowerCase().includes(needle))
      : [...here];
    const by = (a: Entry, b: Entry) => {
      const name = a.name.localeCompare(b.name, "en", {
        numeric: true,
        sensitivity: "base",
      });
      if (sort.key === "kind")
        return kindOf(a).localeCompare(kindOf(b)) || name;
      if (sort.key === "size") return bytesOf(a.size) - bytesOf(b.size) || name;
      if (sort.key === "modified")
        return (
          (new Date(a.modified ?? 0).getTime() || 0) -
            (new Date(b.modified ?? 0).getTime() || 0) || name
        );
      return name;
    };
    // Folders stay on top, like Finder's "Keep folders on top"
    return l.sort(
      (a, b) =>
        Number(isFolder(b)) - Number(isFolder(a)) || by(a, b) * sort.dir,
    );
  })();
  const names = items.map((e) => e.name);
  const current = items.find((e) => e.name === cur) ?? null;
  const optId = (n: string) => `${uid}-${names.indexOf(n)}`;

  const announce = (text: string) => {
    setSaid("");
    window.setTimeout(() => setSaid(text), 30);
  };
  const go = (
    next: string[],
    how: "push" | "back" | "fwd" = "push",
    focusOn?: string,
  ) => {
    if (!listAt(next)) return;
    setHist((h) =>
      how === "push"
        ? { back: [...h.back, path], fwd: [] }
        : how === "back"
          ? { back: h.back.slice(0, -1), fwd: [...h.fwd, path] }
          : { back: [...h.back, path], fwd: h.fwd.slice(0, -1) },
    );
    setPath(next);
    setQ("");
    setLook(false);
    setSel(focusOn ? [focusOn] : []);
    setCur(focusOn ?? null);
    anchor.current = focusOn ?? null;
    const n = listAt(next)!.length;
    announce(`${next.at(-1) ?? name}, ${n === 1 ? "1 item" : `${n} items`}`);
  };
  const goBack = () =>
    hist.back.length &&
    go(
      hist.back[hist.back.length - 1],
      "back",
      path.length > hist.back[hist.back.length - 1].length
        ? path[hist.back[hist.back.length - 1].length]
        : undefined,
    );
  const goFwd = () =>
    hist.fwd.length && go(hist.fwd[hist.fwd.length - 1], "fwd");
  const goUp = () =>
    path.length && go(path.slice(0, -1), "push", path[path.length - 1]);
  const open = (e: Entry) =>
    isFolder(e)
      ? inside(e)
        ? go([...path, e.name])
        : announce(`${e.name} is empty`)
      : onOpen(e.name);
  /** A subfolder in its own tab: it takes its entries along */
  const openInTab = (e: Entry) => {
    if (isFolder(e)) {
      const sub = inside(e) ?? [];
      onSeed(e.name, {
        updated: sub.length === 1 ? "1 item" : `${sub.length} items`,
        files: sub,
      });
    }
    onOpen(e.name);
  };

  /* Changes to the folder – here, at the path you're in */
  const editHere = (fn: (l: Entry[]) => Entry[], say: string) => {
    const at = (list: Entry[], p: string[]): Entry[] =>
      p.length
        ? list.map((e) =>
            e.name === p[0]
              ? { ...e, files: at(inside(e) ?? [], p.slice(1)) }
              : e,
          )
        : fn(list);
    onChange(at(entries.map(entryOf), path));
    announce(say);
  };
  const free = (want: string) => {
    const taken = new Set(here.map((e) => e.name.toLowerCase()));
    if (!taken.has(want.toLowerCase())) return want;
    const dot = want.lastIndexOf(".") > 0 ? want.lastIndexOf(".") : want.length;
    for (let n = 2; ; n++) {
      const next = `${want.slice(0, dot)} ${n}${want.slice(dot)}`;
      if (!taken.has(next.toLowerCase())) return next;
    }
  };
  const targets = () => (sel.length ? sel : cur ? [cur] : []);
  const renameOnce = useRef<string | null>(null);
  const rename = (from: string, to: string) => {
    if (renameOnce.current !== from) return;
    renameOnce.current = null;
    setRenaming(null);
    const next = to.trim();
    window.setTimeout(() => listRef.current?.focus({ preventScroll: true }));
    if (!next || next === from) return;
    if (
      here.some(
        (e) => e.name.toLowerCase() === next.toLowerCase() && e.name !== from,
      )
    )
      return announce(`“${next}” is already taken`);
    const a = lookup(from);
    if (a) onSeed(next, a);
    editHere(
      (l) =>
        l.map((e) =>
          e.name === from ? { ...e, name: next, modified: Date.now() } : e,
        ),
      `Renamed to ${next}`,
    );
    setSel([next]);
    setCur(next);
    anchor.current = next;
  };
  const duplicate = () => {
    const picked = targets();
    if (!picked.length) return;
    const made: string[] = [];
    editHere(
      (l) => {
        const out = [...l];
        for (const n of picked) {
          const e = l.find((x) => x.name === n);
          if (!e) continue;
          const dot =
            n.lastIndexOf(".") > 0 && !isFolder(e)
              ? n.lastIndexOf(".")
              : n.length;
          const copy = free(`${n.slice(0, dot)} copy${n.slice(dot)}`);
          const a = lookup(n);
          if (a) onSeed(copy, a);
          out.push({ ...e, name: copy, modified: Date.now() });
          made.push(copy);
        }
        return out;
      },
      picked.length === 1
        ? `Duplicated ${picked[0]}`
        : `Duplicated ${picked.length} items`,
    );
    setSel(made);
    setCur(made[made.length - 1] ?? null);
  };
  const trash = () => {
    const picked = targets();
    if (!picked.length) return;
    const at = names.indexOf(picked[0]);
    const rest = names.filter((n) => !picked.includes(n));
    editHere(
      (l) => l.filter((e) => !picked.includes(e.name)),
      picked.length === 1
        ? `Moved ${picked[0]} to the trash`
        : `Moved ${picked.length} items to the trash`,
    );
    const next = rest[Math.min(at, rest.length - 1)] ?? null;
    setSel(next ? [next] : []);
    setCur(next);
    setLook(false);
  };
  const newFolder = () => {
    const n = free("untitled folder");
    editHere(
      (l) => [
        ...l,
        { name: n, kind: "Folder", files: [], modified: Date.now() },
      ],
      `New folder ${n}`,
    );
    setQ("");
    setSel([n]);
    setCur(n);
    anchor.current = n;
    setRenaming(n);
  };
  const copyNames = () => {
    const picked = targets();
    if (!picked.length) return;
    toClipboard(picked.join("\n"));
    announce(
      picked.length === 1
        ? `Copied “${picked[0]}”`
        : `Copied ${picked.length} names`,
    );
  };

  const pick = (
    n: string,
    how: { shiftKey?: boolean; metaKey?: boolean; ctrlKey?: boolean },
  ) => {
    if (how.metaKey || how.ctrlKey) {
      setSel((s) => (s.includes(n) ? s.filter((x) => x !== n) : [...s, n]));
      anchor.current = n;
    } else if (
      how.shiftKey &&
      anchor.current &&
      names.includes(anchor.current)
    ) {
      const a = names.indexOf(anchor.current);
      const z = names.indexOf(n);
      setSel(names.slice(Math.min(a, z), Math.max(a, z) + 1));
    } else {
      setSel([n]);
      anchor.current = n;
    }
    setCur(n);
  };
  const moveBy = (step: number, extend: boolean) => {
    if (!names.length) return;
    const at = cur ? names.indexOf(cur) : -1;
    const to =
      names[
        clamp(
          at < 0 ? (step > 0 ? 0 : names.length - 1) : at + step,
          0,
          names.length - 1,
        )
      ];
    pick(to, { shiftKey: extend });
    if (!extend) anchor.current = to;
  };
  /** How many tiles a row of the icon view holds – ↑ ↓ jump by that */
  const perRow = () => {
    const tiles = [...(listRef.current?.children ?? [])] as HTMLElement[];
    const top = tiles[0]?.offsetTop;
    const n = tiles.findIndex((t) => t.offsetTop !== top);
    return n < 0 ? tiles.length || 1 : n;
  };
  useEffect(() => {
    if (cur)
      document
        .getElementById(optId(cur))
        ?.scrollIntoView({ block: "nearest", inline: "nearest" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cur]);
  // Quick Look closes when you click anywhere else
  useEffect(() => {
    if (!look) return;
    const down = (e: PointerEvent) => {
      const t = e.target as Element;
      if (!t.closest(".ui-asw-ql") && !t.closest(`#${CSS.escape(uid)}-list`))
        setLook(false);
    };
    document.addEventListener("pointerdown", down);
    return () => document.removeEventListener("pointerdown", down);
  }, [look, uid]);

  const onListKey = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    if ((e.target as HTMLElement).closest("input")) return;
    const mod = e.metaKey || e.ctrlKey;
    const k = e.key;
    if (isMenuKey(e)) {
      e.preventDefault();
      const el = cur ? document.getElementById(optId(cur)) : null;
      return ctx.openOn(
        el ?? listRef.current,
        el ? "item" : "space",
        listRef.current,
      );
    }
    if (k === "F2" && current) {
      e.preventDefault();
      return setRenaming(current.name);
    }
    if (mod && !e.shiftKey && k.toLowerCase() === "d") {
      e.preventDefault();
      return duplicate();
    }
    if (mod && k === "Backspace") {
      e.preventDefault();
      return trash();
    }
    if (mod && !e.shiftKey && k.toLowerCase() === "c") {
      e.preventDefault();
      return copyNames();
    }
    if (mod && e.shiftKey && k.toLowerCase() === "n") {
      e.preventDefault();
      return newFolder();
    }
    if (mod && (k === "1" || k === "2")) {
      e.preventDefault();
      return setView(k === "1" ? "list" : "icons");
    }
    if (k === " ") {
      e.preventDefault();
      if (!current) return;
      setLook((l) => !l);
      return announce(
        look ? "Quick Look closed" : `Quick Look: ${current.name}`,
      );
    }
    if (k === "Escape" && look) {
      e.preventDefault();
      e.stopPropagation();
      setLook(false);
      return announce("Quick Look closed");
    }
    if (
      k === "Enter" ||
      (mod && k === "ArrowDown") ||
      (mod && k.toLowerCase() === "o")
    ) {
      e.preventDefault();
      if (current) open(current);
      return;
    }
    if (mod && k === "ArrowUp") {
      e.preventDefault();
      return goUp();
    }
    if (mod && (k === "[" || k === "]")) {
      e.preventDefault();
      return k === "[" ? goBack() : goFwd();
    }
    if (mod && k.toLowerCase() === "a") {
      e.preventDefault();
      setSel(names);
      return announce(`${names.length} selected`);
    }
    if (mod && k.toLowerCase() === "f") {
      e.preventDefault();
      return searchRef.current?.focus();
    }
    const grid = view === "icons";
    const step = (
      {
        ArrowDown: grid ? perRow() : 1,
        ArrowUp: grid ? -perRow() : -1,
        ...(grid ? { ArrowRight: 1, ArrowLeft: -1 } : {}),
      } as Record<string, number>
    )[k];
    if (step) {
      e.preventDefault();
      return moveBy(step, e.shiftKey);
    }
    if (k === "Home" || k === "End") {
      e.preventDefault();
      return moveBy(k === "Home" ? -names.length : names.length, e.shiftKey);
    }
    // Type the start of a name to jump to it
    if (k.length === 1 && !mod && !e.altKey && k !== " ") {
      const t =
        Date.now() - typed.current.at > 700 ? k : typed.current.text + k;
      typed.current = { text: t, at: Date.now() };
      const hit = names.find((n) =>
        n.toLowerCase().startsWith(t.toLowerCase()),
      );
      if (hit) pick(hit, {});
    }
  };

  const showDate = width >= 260;
  const shortDate = width < 420;
  const sortItems: MenuItem[] = [
    ...SORTS.map((s) => ({
      label: s.label,
      icon: s.icon,
      checked: sort.key === s.key,
      onSelect: () => setSort((x) => ({ ...x, key: s.key })),
    })),
    {
      label: "Ascending",
      icon: SortByUp01Icon,
      checked: sort.dir === 1,
      divider: true,
      onSelect: () => setSort((x) => ({ ...x, dir: 1 })),
    },
    {
      label: "Descending",
      icon: SortByDown01Icon,
      checked: sort.dir === -1,
      onSelect: () => setSort((x) => ({ ...x, dir: -1 })),
    },
  ];
  const head = (key: SortKey, label: string) => (
    <button
      type="button"
      className="ui-asw-fd__sort"
      data-on={sort.key === key || undefined}
      aria-label={`${label}${sort.key === key ? `, sorted ${sort.dir === 1 ? "ascending" : "descending"}` : ""} – sort by ${label.toLowerCase()}`}
      onClick={() =>
        setSort((x) =>
          x.key === key ? { key, dir: x.dir === 1 ? -1 : 1 } : { key, dir: 1 },
        )
      }
    >
      {label}
      {sort.key === key && (
        <Icon
          icon={sort.dir === 1 ? ArrowUp01Icon : ArrowDown01Icon}
          size={12}
        />
      )}
    </button>
  );
  const crumbs = [name, ...path];
  const count = items.length;

  /* Right-click: on an item (selected first, unless it already is) or on the empty space around them – like Finder */
  const ctx = useContextMenu<"item" | "space">();
  const sortMenu: CtxItem[] = sortItems.map((i) => ({
    label: i.label,
    icon: i.icon!,
    checked: i.checked,
    divider: i.divider,
    onSelect: i.onSelect,
  }));
  const folderMenu = (
    kind: "item" | "space",
  ): { label: string; items: CtxItem[] } => {
    const picked = targets();
    const one =
      picked.length === 1 ? items.find((e) => e.name === picked[0]) : undefined;
    if (kind === "item" && picked.length)
      return {
        label: one?.name ?? `${picked.length} items`,
        items: [
          {
            label: "Open",
            icon: FolderOpenIcon,
            keys: "↵",
            disabled: !one,
            onSelect: () => one && open(one),
          },
          {
            label: "Open in new tab",
            icon: ArrowUpRight01Icon,
            disabled: !one,
            onSelect: () => one && openInTab(one),
          },
          {
            label: "Quick Look",
            icon: ViewIcon,
            keys: "Space",
            disabled: !one,
            onSelect: () => one && (setCur(one.name), setLook(true)),
          },
          {
            label: picked.length === 1 ? "Copy name" : "Copy names",
            icon: Copy01Icon,
            keys: `${MOD}C`,
            divider: true,
            onSelect: copyNames,
          },
          {
            label: "Rename",
            icon: PencilEdit01Icon,
            keys: "F2",
            disabled: !one,
            onSelect: () => one && setRenaming(one.name),
          },
          {
            label: "Duplicate",
            icon: Copy02Icon,
            keys: `${MOD}D`,
            onSelect: duplicate,
          },
          {
            label: "Move to trash",
            icon: Delete02Icon,
            keys: `${MOD}⌫`,
            danger: true,
            divider: true,
            onSelect: trash,
          },
        ],
      };
    return {
      label: crumbs[crumbs.length - 1],
      items: [
        {
          label: "New folder",
          icon: FolderAddIcon,
          keys: `${SHIFT}${MOD}N`,
          onSelect: newFolder,
        },
        {
          label: "View as list",
          icon: ListViewIcon,
          keys: `${MOD}1`,
          checked: view === "list",
          divider: true,
          onSelect: () => setView("list"),
        },
        {
          label: "View as icons",
          icon: GridViewIcon,
          keys: `${MOD}2`,
          checked: view === "icons",
          onSelect: () => setView("icons"),
        },
        { label: "Sort by", icon: Sorting05Icon, submenu: sortMenu },
        {
          label: "Select all",
          icon: CursorTextIcon,
          keys: `${MOD}A`,
          divider: true,
          disabled: !names.length,
          onSelect: () => (setSel(names), announce(`${names.length} selected`)),
        },
      ],
    };
  };

  return (
    <div className="ui-asw-fd">
      <Toolbar label="Folder" className="ui-asw-fd__tools">
        <span className="ui-asw-tb__group">
          <Tool
            icon={ArrowLeft01Icon}
            label="Back"
            keys={`${MOD}[`}
            disabled={!hist.back.length}
            onClick={goBack}
          />
          <Tool
            icon={ArrowRight01Icon}
            label="Forward"
            keys={`${MOD}]`}
            disabled={!hist.fwd.length}
            onClick={goFwd}
          />
        </span>
        <span className="ui-asw-fd__crumbs">
          {crumbs.map((c, i) => (
            <Fragment key={`${i}-${c}`}>
              {i > 0 && (
                <span className="ui-asw-fd__sep" aria-hidden="true">
                  <Icon icon={ArrowRight01Icon} size={12} />
                </span>
              )}
              <button
                type="button"
                className="ui-asw-fd__crumb"
                data-tool
                data-tip={i === crumbs.length - 1 ? undefined : `Go to ${c}`}
                aria-current={i === crumbs.length - 1 ? "location" : undefined}
                title={c}
                onMouseDown={keep}
                onClick={() =>
                  i < crumbs.length - 1 && go(path.slice(0, i), "push", path[i])
                }
              >
                {c}
              </button>
            </Fragment>
          ))}
        </span>
        <span className="ui-asw-tb__group">
          <Tool
            icon={ListViewIcon}
            label="As list"
            pressed={view === "list"}
            onClick={() => setView("list")}
          />
          <Tool
            icon={GridViewIcon}
            label="As icons"
            pressed={view === "icons"}
            onClick={() => setView("icons")}
          />
          <MenuTool
            icon={Sorting05Icon}
            label="Sort by"
            items={sortItems}
            align="end"
          />
        </span>
      </Toolbar>
      <label className="ui-asw-fd__search">
        <Icon icon={Search01Icon} size={14} />
        <input
          ref={searchRef}
          type="search"
          value={q}
          placeholder={`Search ${crumbs[crumbs.length - 1]}`}
          aria-label={`Search ${crumbs[crumbs.length - 1]}`}
          aria-controls={`${uid}-list`}
          spellCheck={false}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Escape" && q) {
              e.preventDefault();
              e.stopPropagation();
              setQ("");
            } else if (e.key === "ArrowDown" || e.key === "Enter") {
              e.preventDefault();
              if (!cur && names[0]) pick(names[0], {});
              listRef.current?.focus();
            }
          }}
        />
      </label>
      {view === "list" && (
        <div className="ui-asw-fd__head" data-date={showDate || undefined}>
          {head("name", "Name")}
          {head("size", "Size")}
          {showDate && head("modified", "Modified")}
        </div>
      )}
      <div
        className="ui-asw-fd__body"
        onPointerDown={(e) => {
          if (e.target === e.currentTarget) {
            setSel([]);
            setCur(null);
          }
        }}
        onContextMenu={(e) => {
          if ((e.target as HTMLElement).closest("input")) return;
          e.preventDefault();
          if ((e.target as HTMLElement).closest(".ui-asw-ql")) return;
          const row = (e.target as HTMLElement).closest<HTMLElement>(
            "[role='option']",
          );
          const n = row ? names[Number(row.id.split("-").pop())] : undefined;
          if (n && !sel.includes(n)) {
            setSel([n]);
            anchor.current = n;
          }
          if (n) setCur(n);
          else {
            setSel([]);
            setCur(null);
          }
          listRef.current?.focus({ preventScroll: true });
          ctx.open(e.clientX, e.clientY, n ? "item" : "space", listRef.current);
        }}
      >
        <div
          ref={listRef}
          id={`${uid}-list`}
          className={view === "list" ? "ui-asw-fd__list" : "ui-asw-fd__grid"}
          role="listbox"
          tabIndex={0}
          aria-label={`${crumbs[crumbs.length - 1]}, ${count === 1 ? "1 item" : `${count} items`}`}
          aria-multiselectable="true"
          aria-orientation={view === "icons" ? "horizontal" : "vertical"}
          aria-activedescendant={
            cur && names.includes(cur) ? optId(cur) : undefined
          }
          onKeyDown={onListKey}
          onFocus={() => !cur && names[0] && setCur(names[0])}
          onPointerDown={(e) => {
            if (e.target === e.currentTarget) {
              setSel([]);
              setCur(null);
            }
          }}
        >
          {items.map((e) => {
            const folder = isFolder(e);
            const sub = folder ? inside(e) : undefined;
            return (
              <div
                key={e.name}
                id={optId(e.name)}
                role="option"
                aria-selected={sel.includes(e.name)}
                aria-label={`${e.name}, ${kindOf(e)}${folder ? "" : `, ${sizeText(e)}`}`}
                className={
                  view === "list" ? "ui-asw-fd__row" : "ui-asw-fd__tile"
                }
                data-cur={cur === e.name || undefined}
                data-date={showDate || undefined}
                onPointerDown={(ev) => {
                  if (ev.button !== 0) return;
                  pick(e.name, ev);
                }}
                onDoubleClick={() => open(e)}
              >
                <span className="ui-asw-fd__name">
                  {fileGlyph(folder ? "" : e.name)}
                  {renaming === e.name ? (
                    <input
                      className="ui-asw-fd__rename"
                      defaultValue={e.name}
                      aria-label={`Rename ${e.name}`}
                      spellCheck={false}
                      autoFocus
                      onFocus={(ev) =>
                        (renameOnce.current = e.name) &&
                        ev.currentTarget.setSelectionRange(
                          0,
                          folder || e.name.lastIndexOf(".") <= 0
                            ? e.name.length
                            : e.name.lastIndexOf("."),
                        )
                      }
                      onPointerDown={(ev) => ev.stopPropagation()}
                      onDoubleClick={(ev) => ev.stopPropagation()}
                      onKeyDown={(ev) => {
                        ev.stopPropagation();
                        if (ev.key === "Enter") {
                          ev.preventDefault();
                          rename(e.name, ev.currentTarget.value);
                        } else if (ev.key === "Escape") {
                          ev.preventDefault();
                          renameOnce.current = null;
                          setRenaming(null);
                          listRef.current?.focus({ preventScroll: true });
                        }
                      }}
                      onBlur={(ev) =>
                        renaming === e.name &&
                        rename(e.name, ev.currentTarget.value)
                      }
                    />
                  ) : (
                    <span dir="auto" title={e.name}>
                      {e.name}
                    </span>
                  )}
                </span>
                {view === "list" && (
                  <span className="ui-asw-fd__meta">
                    {sizeText(e, sub?.length)}
                  </span>
                )}
                {view === "list" && showDate && (
                  <span className="ui-asw-fd__meta">
                    {whenText(e.modified, now, shortDate)}
                  </span>
                )}
              </div>
            );
          })}
        </div>
        {ctx.render(folderMenu)}
        {count === 0 && (
          <p className="ui-asw-fd__empty">
            {q
              ? `Nothing in ${crumbs[crumbs.length - 1]} matches “${q}”`
              : "This folder is empty"}
          </p>
        )}
        {look && current && (
          <QuickLook
            entry={current}
            artifact={isFolder(current) ? undefined : lookup(current.name)}
            inside={isFolder(current) ? inside(current) : undefined}
            onClose={() => {
              setLook(false);
              listRef.current?.focus();
            }}
          />
        )}
      </div>
      <p className="ui-asw-fd__foot ui-asw-num">
        {count === 1 ? "1 item" : `${number0.format(count)} items`}
        {sel.length > 0 && `, ${number0.format(sel.length)} selected`}
      </p>
      <span className="ui-asw-sr" role="status" aria-live="polite">
        {said}
      </span>
    </div>
  );
}

/** Quick Look: a look into the selected file without opening it – Space or Esc closes it */
function QuickLook({
  entry,
  artifact,
  inside,
  onClose,
}: {
  entry: Entry;
  artifact?: ShellArtifact;
  inside?: Entry[];
  onClose: () => void;
}) {
  const a = artifact;
  const table = a?.table;
  const ev = useMemo(
    () => (table ? makeEval((r, c) => table[r]?.[c] ?? "") : null),
    [table],
  );
  return (
    <div
      className="ui-asw-ql"
      role="dialog"
      aria-label={`Quick Look: ${entry.name}`}
    >
      <header className="ui-asw-ql__head">
        <span className="ui-asw-art__glyph" aria-hidden="true">
          {fileGlyph(isFolder(entry) ? "" : entry.name)}
        </span>
        <span className="ui-asw-art__title">
          <b dir="auto" title={entry.name}>
            {entry.name}
          </b>
          <span>
            {kindOf(entry)}
            <i aria-hidden="true"> · </i>
            {sizeText(entry, inside?.length)}
          </span>
        </span>
        <button
          type="button"
          className="ui-asw-ql__close"
          aria-label="Close Quick Look"
          onClick={onClose}
        >
          <Icon icon={Cancel01Icon} size={12} stroke={2.2} />
        </button>
      </header>
      <div className="ui-asw-ql__body">
        {inside ? (
          <ul className="ui-asw-art__files">
            {inside.map((e) => (
              <li key={e.name}>
                {fileGlyph(isFolder(e) ? "" : e.name)}
                <span dir="auto">{e.name}</span>
              </li>
            ))}
          </ul>
        ) : a?.preview ? (
          a.preview
        ) : a?.doc ? (
          <div
            className="ui-asw-doc ui-asw-doc--still"
            dangerouslySetInnerHTML={{ __html: docHtml(a.doc) }}
          />
        ) : table && ev ? (
          <div className="ui-asw-ql__sheet">
            <table>
              <tbody>
                {table.slice(0, 8).map((row, r) => (
                  <tr key={r} data-head={r === 0 || undefined}>
                    {row.slice(0, 6).map((v, c) => (
                      <td
                        key={c}
                        data-num={typeof ev(r, c) === "number" || undefined}
                      >
                        {shown(
                          v,
                          ev(r, c),
                          a.sheet?.styles?.[`${r}:${c}`]?.format,
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : a?.code ? (
          <pre className="ui-asw-art__code">{a.code}</pre>
        ) : a?.image ? (
          <Wireframe blocks={a.image} />
        ) : (
          <p className="ui-asw-ql__none">No preview</p>
        )}
      </div>
    </div>
  );
}

/* ---------- Styles ---------- */

/* ---------- Workspace: the work area's default content ---------- */

export type WorkspaceTool = {
  id: string;
  label: string;
  icon: Parameters<typeof Icon>[0]["icon"];
  shortcut?: string;
};
export type WorkspaceSource = {
  id: string;
  title: string;
  source: string;
  mark?: ReactNode;
  /** In a New tab of the workspace: opens this artifact (by file name) in that tab */
  artifact?: string;
};

export const defaultTools: WorkspaceTool[] = [
  { id: "terminal", label: "Terminal", icon: CommandLineIcon, shortcut: "⌃`" },
  { id: "browser", label: "Browser", icon: Globe02Icon, shortcut: "⌘⇧B" },
  { id: "notes", label: "Notes", icon: NoteEditIcon, shortcut: "⌘⇧N" },
];

export const defaultSources: WorkspaceSource[] = [
  {
    id: "geo",
    title: "GEO: Generative Engine Optimization",
    source: "arxiv.org",
    mark: "A",
  },
  {
    id: "tokens",
    title: "Design tokens – naming guide",
    source: "notion.so",
    mark: "N",
  },
  { id: "q3", title: "Q3 launch checklist", source: "linear.app", mark: "L" },
];

/** Sample artifacts for the work area – what the suggestions in sampleSources open in a New tab */
export const sampleArtifacts: Record<string, ShellArtifact> = {
  "q3-launch-checklist.md": {
    updated: "Updated 2h ago",
    doc: {
      title: "Q3 launch checklist",
      paragraphs: [
        "Everything that has to be true before the launch on Sep 30. Tick it off as it lands.",
      ],
      bullets: [
        "[x] Pricing page copy final",
        "[x] Changelog drafted",
        "[ ] Onboarding emails scheduled",
        "[ ] Support macros updated",
        "[ ] Launch post reviewed",
      ],
    },
  },
  "q3-launch-budget.csv": {
    updated: "Updated yesterday",
    table: [
      ["Channel", "Weekly", "Weeks", "Total"],
      ["Search", "1,200", "6", "=B2*C2"],
      ["Social", "800", "6", "=B3*C3"],
      ["Newsletter", "300", "4", "=B4*C4"],
      ["Events", "2,500", "1", "=B5*C5"],
      ["Total", "=SUM(B2:B5)", "", "=SUM(D2:D5)"],
    ],
    sheet: {
      styles: {
        "5:0": { bold: true },
        "1:3": { format: "currency" },
        "2:3": { format: "currency" },
        "3:3": { format: "currency" },
        "4:3": { format: "currency" },
        "5:3": { format: "currency", bold: true },
      },
    },
  },
  "design-tokens-naming.md": {
    updated: "Updated Mon",
    doc: {
      title: "Design tokens – naming guide",
      paragraphs: [
        "Tokens are named for what they do, not for how they look: color-text-muted, not gray-500. A name reads from the general to the specific.",
      ],
      bullets: [
        "Category first: color, space, radius, shadow",
        "Then the role: text, surface, line",
        "State last: hover, pressed, disabled",
      ],
    },
  },
};

/** Suggestions for a New tab: the first opens a web page (onOpen), the others open sampleArtifacts in the tab */
export const sampleSources: WorkspaceSource[] = [
  defaultSources[0],
  { ...defaultSources[1], artifact: "design-tokens-naming.md" },
  { ...defaultSources[2], artifact: "q3-launch-checklist.md" },
  {
    id: "budget",
    title: "Q3 launch budget",
    source: "sheets.google.com",
    mark: "S",
    artifact: "q3-launch-budget.csv",
  },
];

export type WorkspacePanelProps = {
  tools?: WorkspaceTool[];
  sources?: WorkspaceSource[];
  /** A tool or a suggestion was opened */
  onOpen?: (id: string) => void;
};

/** The pane's content: an address field, the tools, and suggested sources – each row opens its item */
export function WorkspacePanel({
  tools = defaultTools,
  sources = defaultSources,
  onOpen,
}: WorkspacePanelProps) {
  const [query, setQuery] = useState("");
  return (
    <div className="ui-asw">
      <style href="ui-app-shell-workspace" precedence="default">
        {workspaceCss}
      </style>
      <label className="ui-asw__address">
        <Icon icon={Search01Icon} size={14} />
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search or enter URL"
          aria-label="Search or enter URL"
        />
      </label>

      <section className="ui-asw__section" aria-label="Tools">
        <h3 className="ui-asw__label">Tools</h3>
        {tools.length === 0 && (
          <p className="ui-asw__empty">No tools connected yet</p>
        )}
        {tools.length > 0 && (
          <div className="ui-asw__list">
            {tools.map((tool) => (
              <div key={tool.id} className="ui-asw__row">
                <button
                  type="button"
                  className="ui-asw__open"
                  onClick={() => onOpen?.(tool.id)}
                >
                  <span className="ui-asw__tile">
                    <Icon icon={tool.icon} size={15} />
                  </span>
                  <span className="ui-asw__name" title={tool.label}>
                    {tool.label}
                  </span>
                  {tool.shortcut && (
                    <kbd className="ui-asw__kbd">{tool.shortcut}</kbd>
                  )}
                </button>
                <button
                  type="button"
                  className="ui-asw__more"
                  aria-label={`More for ${tool.label}`}
                >
                  <Icon icon={MoreHorizontalIcon} size={15} />
                </button>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="ui-asw__section" aria-label="Suggested">
        <h3 className="ui-asw__label">Suggested</h3>
        {sources.length === 0 && (
          <p className="ui-asw__empty">Nothing suggested right now</p>
        )}
        {sources.length > 0 && (
          <div className="ui-asw__list">
            {sources.map((item) => (
              <button
                key={item.id}
                type="button"
                className="ui-asw__row ui-asw__open ui-asw__source"
                onClick={() => onOpen?.(item.id)}
              >
                <span className="ui-asw__tile ui-asw__tile--mark">
                  {item.mark ?? firstChar(item.title).toUpperCase()}
                </span>
                <span className="ui-asw__text">
                  <span className="ui-asw__name" title={item.title}>
                    {item.title}
                  </span>
                  <span className="ui-asw__meta" title={item.source}>
                    {item.source}
                  </span>
                </span>
                <span className="ui-asw__go">
                  <Icon icon={ArrowUpRight01Icon} size={14} />
                </span>
              </button>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

const artifactCss = /* css */ `
/* Workspace artifacts: the shared viewers in the work area's tabs – document, sheet, folder, code, image, menus */
.ui-as {
  --asw-out: cubic-bezier(0.23, 1, 0.32, 1);
  --asw-wire-ink: rgba(0, 0, 0, 0.42);
  --asw-wire-mid: rgba(0, 0, 0, 0.16);
  --asw-wire-soft: rgba(0, 0, 0, 0.07);
  --asw-grid: rgba(0, 0, 0, 0.08);
  --asw-rim: rgba(0, 0, 0, 0.025);
  /* Selection in a sheet or a folder – the only place the accent tints anything */
  --asw-sel-bg: rgba(0, 96, 226, 0.07);
  --asw-sel-head: rgba(0, 96, 226, 0.1);
  --asw-err: var(--aui-danger, #c4320a);
  /* File glyphs by type: documents blue, sheets green, code orange, folders amber, images pink */
  --asw-file-doc: var(--aui-blue, #0190fe);
  --asw-file-sheet: var(--aui-green, #5fc88a);
  --asw-file-code: var(--aui-orange, #f28a2e);
  --asw-file-folder: var(--aui-amber, #e8b000);
  --asw-file-image: var(--aui-pink, #ee3e9e);
}
:where(.dark, [data-theme="dark"]) .ui-as {
  --asw-wire-ink: rgba(255, 255, 255, 0.5);
  --asw-wire-mid: rgba(255, 255, 255, 0.2);
  --asw-wire-soft: rgba(255, 255, 255, 0.08);
  --asw-grid: rgba(255, 255, 255, 0.09);
  --asw-rim: rgba(255, 255, 255, 0.03);
  --asw-sel-bg: color-mix(in srgb, var(--aui-blue, #4fb0fe) 13%, transparent);
  --asw-sel-head: color-mix(in srgb, var(--aui-blue, #4fb0fe) 20%, transparent);
  --asw-err: var(--aui-red, #ff6b5f);
  --asw-file-doc: var(--aui-blue, #4fb0fe);
  --asw-file-sheet: var(--aui-green, #7dcf9f);
  --asw-file-code: var(--aui-orange, #f2a15c);
  --asw-file-folder: var(--aui-amber, #e6c24d);
  --asw-file-image: var(--aui-pink, #f06db5);
}
.ui-asw-glyph[data-type] { color: var(--asw-file-doc); }
.ui-asw-glyph[data-type="sheet"] { color: var(--asw-file-sheet); }
.ui-asw-glyph[data-type="code"] { color: var(--asw-file-code); }
.ui-asw-glyph[data-type="folder"] { color: var(--asw-file-folder); }
.ui-asw-glyph[data-type="image"] { color: var(--asw-file-image); }
.ui-asw-num { font-variant-numeric: tabular-nums; }
.ui-asw-sr { position: absolute; width: 1px; height: 1px; margin: -1px; padding: 0; overflow: hidden; clip-path: inset(50%); white-space: nowrap; border: 0; }
/* A tab's content: it fills the white sheet, and nothing in it widens the work area */
.ui-asw-pane { display: grid; grid-template-columns: minmax(0, 1fr); min-height: 100%; }
.ui-asw-pane[hidden] { display: none; }
/* Small buttons in the viewers, e.g. the link field's */
.ui-asw-btn { display: inline-flex; align-items: center; justify-content: center; gap: 6px; height: 28px; padding: 0 10px; border-radius: calc(var(--aui-radius, 7px) - 1px); background: var(--card); box-shadow: var(--card-shadow); font-size: 13px; font-weight: 500; color: var(--ink); white-space: nowrap; transition: box-shadow 140ms ease, background-color 140ms ease; }
.ui-asw-btn:hover { box-shadow: var(--card-shadow-hover); }
.ui-asw-btn--quiet { background: none; box-shadow: none; color: var(--text); }
.ui-asw-btn--quiet:hover { background: var(--chip); box-shadow: none; color: var(--ink); }
.ui-asw-btn:active { transform: scale(0.97); }
/* Images as gray blocks */
.ui-asw-wire { display: block; width: 100%; height: auto; }
.ui-asw-wire__frame { fill: var(--card); stroke: none; }
.ui-asw-wire__block[data-tone="ink"] { fill: var(--asw-wire-ink); }
.ui-asw-wire__block[data-tone="mid"] { fill: var(--asw-wire-mid); }
.ui-asw-wire__block[data-tone="soft"] { fill: var(--asw-wire-soft); }
/* Artifacts: soft gray file chips */
.ui-asw-files { display: flex; flex-wrap: wrap; gap: 6px; }
.ui-asw-file { display: inline-flex; align-items: center; gap: 6px; max-width: 100%; height: 26px; padding: 0 9px 0 7px; border-radius: calc(var(--aui-radius-sm, 5px) + 1px); background: var(--chip); font-size: 12px; color: var(--ink); transition: background-color 140ms ease; }
.ui-asw-file:hover { background: linear-gradient(var(--hover), var(--hover)), var(--chip); }
.ui-asw-file > svg { flex: none; color: var(--muted); }
.ui-asw-file__name { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
/* An opened artifact */
.ui-asw-art { display: flex; flex-direction: column; gap: 16px; padding: 2px 4px 16px; }
.ui-asw-art__head { display: flex; align-items: center; gap: 10px; min-width: 0; }
.ui-asw-art__glyph { display: grid; flex: none; place-items: center; width: 28px; height: 28px; border-radius: 7px; background: var(--chip); color: var(--muted); }
.ui-asw-art__glyph svg { width: 14px; height: 14px; }
.ui-asw-art__title { display: grid; min-width: 0; }
.ui-asw-art__title b { overflow: hidden; font-size: 13px; font-weight: 500; line-height: 18px; white-space: nowrap; text-overflow: ellipsis; color: var(--ink); }
.ui-asw-art__title span { overflow: hidden; font-size: 12px; line-height: 16px; white-space: nowrap; text-overflow: ellipsis; color: var(--muted); }
.ui-asw-art__title i { font-style: normal; color: var(--faint); }
/* Content sits right on the white sheet – a document as plain text, a sheet edge to edge */
.ui-asw-art__doc { font-size: 13px; line-height: 20px; color: var(--text); }
.ui-asw-art__doc h3 { margin: 0 0 8px; font-size: 15px; font-weight: 500; line-height: 22px; color: var(--ink); }
.ui-asw-art__doc p { margin: 0 0 10px; }
.ui-asw-art__doc ul { display: grid; gap: 4px; margin: 0; padding: 0; list-style: none; }
.ui-asw-art__doc li { position: relative; padding-left: 16px; }
.ui-asw-art__doc li::before { content: ""; position: absolute; top: 9px; left: 4px; width: 4px; height: 4px; border-radius: 50%; background: var(--faint); }
.ui-asw-art__code { margin: 0; padding: 14px 16px; border-radius: var(--aui-radius-lg, 10px); background: var(--chip); font-family: var(--aui-font-mono, ui-monospace, SFMono-Regular, Menlo, monospace); font-size: 12px; line-height: 19px; color: var(--ink); white-space: pre-wrap; overflow-wrap: anywhere; }
.ui-asw-art__image { padding: 16px; border-radius: var(--aui-radius-lg, 10px); background: var(--card); box-shadow: var(--card-shadow); }
.ui-asw-art__image .ui-asw-wire { display: block; width: 100%; height: auto; }
.ui-asw-art__files { display: grid; margin: 0; padding: 0; list-style: none; }
.ui-asw-art__files li { display: flex; align-items: center; gap: 8px; min-height: 34px; font-size: 13px; color: var(--ink); }
.ui-asw-art__files li + li { box-shadow: inset 0 0.5px 0 var(--line); }
.ui-asw-art__files svg { flex: none; color: var(--muted); }
.ui-asw-art__files span { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.ui-asw-art__none { margin: 0; font-size: 13px; color: var(--muted); }

/* ---------- Artifact toolbars – compact gray tools, groups apart, a hairline under them edge to edge ---------- */
.ui-asw-tb { position: relative; display: flex; flex-wrap: wrap; align-items: center; gap: 4px 8px; margin: 0 -18px; padding: 0 12px 8px; box-shadow: inset 0 -0.5px 0 var(--line); }
.ui-asw-tb__group { display: inline-flex; flex: none; align-items: center; gap: 2px; }
.ui-asw-tool-anchor { flex: none; }
.ui-asw-tool { display: inline-flex; flex: none; align-items: center; justify-content: center; gap: 4px; min-width: 28px; height: 28px; padding: 0; border-radius: calc(var(--aui-radius-sm, 5px) + 1px); color: var(--muted); transition: background-color 120ms ease, color 120ms ease; }
.ui-asw-tool--menu { gap: 2px; padding: 0 4px 0 6px; }
.ui-asw-tool--menu > svg:last-child { color: var(--faint); }
.ui-asw-tool:hover { background: var(--hover); color: var(--ink); }
.ui-asw-tool:is([aria-pressed="true"], [aria-expanded="true"]) { background: var(--chip); color: var(--ink); }
.ui-asw-tool[aria-disabled] { opacity: 0.35; cursor: default; }
.ui-asw-tool[aria-disabled]:hover { background: none; color: var(--muted); }
.ui-asw-tool:active:not([aria-disabled]) { transform: scale(0.96); }
.ui-asw-tool__text { padding: 0 2px 0 4px; font-size: 13px; color: var(--text); white-space: nowrap; }
/* The tip under a tool: dark, small, kept inside the toolbar's width */
.ui-asw-tip { position: absolute; z-index: 40; left: 0; display: flex; gap: 6px; padding: 3px 7px; border-radius: 6px; background: var(--ink); font-size: 12px; line-height: 16px; color: var(--card); white-space: nowrap; pointer-events: none; animation: ui-asw-fade 120ms ease both; }
.ui-asw-tip kbd { font: inherit; opacity: 0.6; }

/* ---------- Document ---------- */
.ui-asw-dc { display: flex; flex-direction: column; min-width: 0; }
/* The tools stay at the top while the page scrolls under them */
.ui-asw-dc__tools { position: sticky; top: -16px; z-index: 4; padding-top: 4px; margin-top: -4px; background: var(--card); }
.ui-asw-doc { min-height: 220px; padding: 16px 0 12px; font-size: 13px; line-height: 20px; color: var(--text); outline: none; overflow-wrap: anywhere; cursor: text; }
.ui-as .ui-asw-doc:focus-visible { outline: none; }
.ui-asw-doc > :first-child { margin-top: 0; }
.ui-asw-doc :is(h1, h2, h3) { color: var(--ink); font-weight: 500; }
.ui-asw-doc h1 { margin: 0 0 8px; font-size: 18px; line-height: 26px; letter-spacing: -0.15px; }
.ui-asw-doc h2 { margin: 18px 0 6px; font-size: 15px; line-height: 22px; }
.ui-asw-doc h3 { margin: 14px 0 4px; font-size: 13px; line-height: 20px; }
.ui-asw-doc p { margin: 0 0 10px; }
.ui-asw-doc blockquote { margin: 0 0 10px; padding-left: 12px; box-shadow: inset 2px 0 0 var(--line-strong); color: var(--muted); }
/* Bold in the text is 600: the browser's own bold commands read the weight, so 500 wouldn't toggle off */
.ui-asw-doc :is(b, strong) { font-weight: 600; color: var(--ink); }
.ui-asw-doc a { color: var(--ink); text-decoration: underline; text-decoration-color: var(--line-strong); text-underline-offset: 2px; }
.ui-asw-doc code { padding: 1px 4px; border-radius: calc(var(--aui-radius-sm, 5px) - 1px); background: var(--chip); font-family: var(--aui-font-mono, ui-monospace, SFMono-Regular, Menlo, monospace); font-size: 12px; color: var(--ink); }
.ui-asw-doc :is(ul, ol) { margin: 0 0 10px; padding-left: 20px; }
.ui-asw-doc :is(ul, ol) :is(ul, ol) { margin: 2px 0 0; }
.ui-asw-doc li { margin: 2px 0; padding-left: 2px; }
.ui-asw-doc li::marker { color: var(--faint); }
/* A checklist: a box in front of each item – click it, or ⌘ Enter; done items go gray */
.ui-asw-doc ul[data-check] { padding-left: 0; list-style: none; }
.ui-asw-doc ul[data-check] > li { position: relative; padding-left: 24px; }
.ui-asw-doc ul[data-check] > li::before { content: ""; position: absolute; top: 3px; left: 1px; width: 14px; height: 14px; border-radius: 4px; background: var(--card); box-shadow: inset 0 0 0 1px var(--line-strong), 0 0.5px 1px rgba(0, 0, 0, 0.06); cursor: pointer; transition: background-color 120ms ease, box-shadow 120ms ease; }
.ui-asw-doc ul[data-check] > li[data-done]::before { background: var(--ink); box-shadow: none; }
.ui-asw-doc ul[data-check] > li[data-done]::after { content: ""; position: absolute; top: 6px; left: 6px; width: 3.5px; height: 6.5px; border: solid var(--card); border-width: 0 1.5px 1.5px 0; transform: rotate(45deg); pointer-events: none; }
.ui-asw-doc ul[data-check] > li[data-done] { color: var(--muted); text-decoration: line-through; text-decoration-color: var(--faint); }
.ui-asw-doc--still { min-height: 0; padding: 0; cursor: auto; }
.ui-asw-dc__foot { margin: 0 -18px; padding: 8px 18px 0; font-size: 12px; line-height: 16px; color: var(--faint); box-shadow: inset 0 0.5px 0 var(--line); }
.ui-asw-dc__foot i { font-style: normal; }
/* The link field */
.ui-asw-linkpop { width: min(280px, 80vw); padding: 6px; border-radius: var(--aui-radius-lg, 10px); background: var(--aui-overlay, var(--card)); box-shadow: var(--aui-shadow-popover, 0 0 0 0.5px var(--line-strong), 0 12px 32px -8px rgba(0, 0, 0, 0.16)); }
.ui-asw-linkpop form { display: flex; align-items: center; gap: 6px; }
.ui-asw-linkpop input { flex: 1; min-width: 0; height: 28px; padding: 0 8px; border: 0; border-radius: calc(var(--aui-radius, 7px) - 1px); background: none; box-shadow: inset 0 0 0 0.5px var(--line-strong); font: inherit; font-size: 13px; color: var(--ink); outline: none; }
.ui-asw-linkpop input::placeholder { color: var(--placeholder); }
/* Typing in a field: no ring – the caret shows where you are, the edge only firms up a little */
.ui-asw-linkpop input:is(:focus, :focus-visible) { outline: none; box-shadow: inset 0 0 0 1px var(--line-strong); }
.ui-asw-linkpop .ui-asw-btn { flex: none; }

/* ---------- Sheet ---------- */
.ui-asw-sh { display: flex; flex-direction: column; min-width: 0; }
/* The name box, fx and the formula bar – one hairline field */
.ui-asw-sh__bar { display: grid; grid-template-columns: 64px 22px minmax(0, 1fr); align-items: center; height: 28px; margin: 10px 0; border-radius: calc(var(--aui-radius, 7px) - 1px); box-shadow: inset 0 0 0 0.5px var(--line-strong); font-size: 12px; transition: box-shadow 120ms ease; }
.ui-asw-sh__bar:focus-within { box-shadow: inset 0 0 0 1px var(--line-strong); }
.ui-asw-sh__bar input { min-width: 0; height: 100%; padding: 0 8px; border: 0; background: none; font: inherit; color: var(--ink); outline: none; }
.ui-asw-sh__ref { box-shadow: inset -0.5px 0 0 var(--line-strong); text-align: center; }
.ui-asw-sh__fx { font-size: 12px; font-style: italic; text-align: center; color: var(--faint); }
.ui-asw-sh__formula { padding-left: 2px !important; }
/* The grid edge to edge – it scrolls sideways inside itself, never the panel */
.ui-asw-sh__wrap { position: relative; margin: 0 -18px; overflow-x: auto; overflow-y: hidden; overscroll-behavior-x: none; background: var(--card); scrollbar-width: thin; }
.ui-asw-sh__grid { table-layout: fixed; border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; outline: none; user-select: none; -webkit-user-select: none; touch-action: manipulation; }
.ui-as .ui-asw-sh__grid:focus-visible { outline: none; }
.ui-asw-sh__grid :is(th, td) { box-sizing: border-box; height: 26px; padding: 0 8px; overflow: hidden; white-space: nowrap; text-overflow: clip; text-align: left; box-shadow: inset -0.5px -0.5px 0 var(--asw-grid); }
.ui-asw-sh__cell { position: relative; color: var(--ink); cursor: cell; }
.ui-asw-sh__cell[data-num] { text-align: right; }
.ui-asw-sh__cell[data-bold] { font-weight: 500; }
.ui-asw-sh__cell[data-err] { color: var(--asw-err); text-align: center; }
.ui-asw-sh__cell[data-sel] { background: var(--asw-sel-bg); }
/* Letters and numbers: a faint gray rim, centered – the selected ones tint */
.ui-asw-sh__grid :is(.ui-asw-sh__corner, .ui-asw-sh__col, .ui-asw-sh__row) { padding: 0; background: var(--asw-rim); font-size: 11px; font-weight: 400; text-align: center; color: var(--faint); cursor: default; transition: background-color 80ms ease, color 80ms ease; }
.ui-asw-sh__grid :is(.ui-asw-sh__col, .ui-asw-sh__corner) { position: relative; height: 22px; overflow: visible; box-shadow: inset -0.5px -0.5px 0 var(--asw-grid), inset 0 0.5px 0 var(--asw-grid); }
.ui-asw-sh__grid :is(.ui-asw-sh__row, .ui-asw-sh__corner) { position: sticky; left: 0; z-index: 2; background: linear-gradient(var(--asw-rim), var(--asw-rim)), var(--card); }
.ui-asw-sh__grid .ui-asw-sh__col[data-on] { background: var(--asw-sel-head); color: var(--ink); }
.ui-asw-sh__grid .ui-asw-sh__row[data-on] { background: linear-gradient(var(--asw-sel-head), var(--asw-sel-head)), var(--card); color: var(--ink); }
.ui-asw-sh__grid .ui-asw-sh__pad { padding: 0; box-shadow: inset 0 -0.5px 0 var(--asw-grid); }
.ui-asw-sh__grid thead .ui-asw-sh__pad { background: var(--asw-rim); box-shadow: inset 0 -0.5px 0 var(--asw-grid), inset 0 0.5px 0 var(--asw-grid); }
/* A column's right edge: drag it, or double-click to fit */
.ui-asw-sh__resize { position: absolute; top: 0; right: -4px; z-index: 3; width: 8px; height: 100%; cursor: col-resize; touch-action: none; }
.ui-asw-sh__resize::after { content: ""; position: absolute; top: 3px; bottom: 3px; left: 3.25px; width: 1.5px; border-radius: 1px; background: var(--accent); opacity: 0; transition: opacity 120ms ease; }
.ui-asw-sh__resize:hover::after { opacity: 1; }
/* The selection's ring with the fill handle – gray while the sheet doesn't have the focus */
.ui-asw-sh__ring { position: absolute; z-index: 1; box-sizing: border-box; box-shadow: inset 0 0 0 2px var(--muted); pointer-events: none; }
.ui-asw-sh:focus-within .ui-asw-sh__ring { box-shadow: inset 0 0 0 2px var(--accent); }
/* A range: a thin ring around it all, the first cell keeps the strong one */
.ui-asw-sh__ring[data-range], .ui-asw-sh:focus-within .ui-asw-sh__ring[data-range] { box-shadow: inset 0 0 0 1px var(--muted); }
.ui-asw-sh:focus-within .ui-asw-sh__ring[data-range] { box-shadow: inset 0 0 0 1px var(--accent); }
.ui-asw-sh__handle { position: absolute; right: -3px; bottom: -3px; width: 6px; height: 6px; background: var(--muted); box-shadow: 0 0 0 1px var(--card); pointer-events: auto; cursor: crosshair; touch-action: none; }
.ui-asw-sh:focus-within .ui-asw-sh__handle { background: var(--accent); }
.ui-asw-sh__fill { position: absolute; z-index: 1; outline: 1px dashed var(--muted); outline-offset: -1px; pointer-events: none; }
/* Writing in a cell: the field sits on it and grows with what you type */
.ui-asw-sh__input { position: absolute; z-index: 3; top: 0; left: 0; box-sizing: border-box; height: 100%; padding: 0 7px; border: 0; background: var(--card); box-shadow: inset 0 0 0 1.5px var(--accent), 0 4px 12px -4px rgba(0, 0, 0, 0.16); font: inherit; color: var(--ink); text-align: left; outline: none; }
/* Sum · Average · Count of the selection, like Excel's status bar */
.ui-asw-sh__status { display: flex; justify-content: flex-end; align-items: center; gap: 14px; height: 28px; margin: 0 -18px; padding: 0 18px; overflow: hidden; font-size: 12px; color: var(--muted); white-space: nowrap; box-shadow: inset 0 0.5px 0 var(--asw-grid); }
.ui-asw-sh__status b { font-weight: 500; color: var(--ink); }

/* ---------- Folder ---------- */
.ui-asw-fd { display: flex; flex-direction: column; min-width: 0; }
.ui-asw-fd__tools { flex-wrap: nowrap; }
.ui-asw-fd__crumbs { display: flex; flex: 1 1 0; align-items: center; min-width: 0; overflow: hidden; }
.ui-asw-fd__crumb { flex: 0 1 auto; min-width: 24px; max-width: 100%; height: 28px; padding: 0 6px; overflow: hidden; border-radius: calc(var(--aui-radius-sm, 5px) + 1px); font-size: 13px; color: var(--muted); white-space: nowrap; text-overflow: ellipsis; transition: background-color 120ms ease, color 120ms ease; }
.ui-asw-fd__crumb:not([aria-current]):hover { background: var(--hover); color: var(--ink); }
.ui-asw-fd__crumb[aria-current] { flex-shrink: 0.2; font-weight: 500; color: var(--ink); cursor: default; }
.ui-asw-fd__sep { display: grid; flex: none; place-items: center; color: var(--faint); }
.ui-asw-fd__search { display: flex; align-items: center; gap: 6px; height: 28px; margin: 10px 0 6px; padding: 0 8px; border-radius: calc(var(--aui-radius, 7px) - 1px); box-shadow: inset 0 0 0 0.5px var(--line-strong); color: var(--faint); transition: box-shadow 120ms ease; }
.ui-asw-fd__search:focus-within { box-shadow: inset 0 0 0 1px var(--line-strong); }
.ui-asw-fd__search > svg { flex: none; }
.ui-asw-fd__search input { flex: 1; min-width: 0; height: 100%; padding: 0; border: 0; background: none; font: inherit; font-size: 13px; color: var(--ink); outline: none; }
.ui-asw-fd__search input::placeholder { color: var(--placeholder); }
/* The list: Name, Size, Modified – fixed rows, hairlines between them */
.ui-asw-fd__head, .ui-asw-fd__row { display: grid; grid-template-columns: minmax(0, 1fr) 64px; column-gap: 10px; align-items: center; padding: 0 8px; }
.ui-asw-fd__head[data-date], .ui-asw-fd__row[data-date] { grid-template-columns: minmax(0, 1fr) 58px 72px; }
@container (min-width: 420px) { .ui-asw-fd__head[data-date], .ui-asw-fd__row[data-date] { grid-template-columns: minmax(0, 1fr) 64px 112px; } }
.ui-asw-fd { container-type: inline-size; }
.ui-asw-fd__head { height: 26px; box-shadow: inset 0 -0.5px 0 var(--line); }
.ui-asw-fd__sort { display: inline-flex; align-items: center; gap: 3px; height: 22px; padding: 0 4px; margin: 0 -4px; border-radius: var(--aui-radius-sm, 5px); font-size: 12px; color: var(--muted); justify-self: start; transition: background-color 120ms ease, color 120ms ease; }
.ui-asw-fd__sort:hover { background: var(--hover); color: var(--ink); }
.ui-asw-fd__sort[data-on] { color: var(--ink); }
.ui-asw-fd__head > .ui-asw-fd__sort:nth-child(2) { justify-self: end; }
/* Free space under the items, like Finder's – a right-click there is about the folder, not an item */
.ui-asw-fd__body { position: relative; min-height: 236px; padding-bottom: 28px; }
.ui-asw-fd__list, .ui-asw-fd__grid { outline: none; }
.ui-as .ui-asw-fd__body [role="listbox"]:focus-visible { outline: none; }
.ui-asw-fd__row { height: 30px; font-size: 13px; color: var(--ink); cursor: default; box-shadow: inset 0 -0.5px 0 var(--line); }
.ui-asw-fd__name { display: flex; align-items: center; gap: 8px; min-width: 0; }
.ui-asw-fd__name > svg { flex: none; width: 14px; height: 14px; color: var(--muted); }
.ui-asw-fd__name > span { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.ui-asw-fd__meta { overflow: hidden; font-size: 12px; color: var(--muted); white-space: nowrap; text-overflow: ellipsis; font-variant-numeric: tabular-nums; }
.ui-asw-fd__row > .ui-asw-fd__meta:nth-child(2) { text-align: right; }
.ui-asw-fd__row > .ui-asw-fd__meta:nth-child(3), .ui-asw-fd__head > .ui-asw-fd__sort:nth-child(3) { justify-self: end; text-align: right; }
.ui-asw-fd__row[aria-selected="true"], .ui-asw-fd__tile[aria-selected="true"] { background: var(--asw-sel-bg); }
/* The keyboard's place in the list: an inset ring, only while the list has the keyboard focus */
.ui-asw-fd__body [role="listbox"]:focus-visible [data-cur] { outline: 2px solid var(--ui-as-focus); outline-offset: -2px; }
.ui-as .ui-asw-fd__body [role="listbox"]:focus-visible:not(:has([data-cur])) { outline: 2px solid var(--ui-as-focus); outline-offset: -2px; }
/* Icons: tiles with the glyph and the name under it */
.ui-asw-fd__grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(84px, 1fr)); gap: 4px; padding: 8px 0 4px; }
.ui-asw-fd__tile { display: flex; flex-direction: column; align-items: center; height: 88px; padding: 10px 4px 6px; border-radius: calc(var(--aui-radius-sm, 5px) + 3px); cursor: default; }
.ui-asw-fd__tile .ui-asw-fd__name { flex-direction: column; gap: 6px; width: 100%; }
.ui-asw-fd__tile .ui-asw-fd__name > svg { width: 36px; height: 36px; }
.ui-asw-fd__tile .ui-asw-fd__name > span { display: -webkit-box; max-width: 100%; font-size: 12px; line-height: 16px; text-align: center; white-space: normal; overflow-wrap: anywhere; -webkit-line-clamp: 2; -webkit-box-orient: vertical; color: var(--ink); }
.ui-asw-fd__empty { margin: 0; padding: 24px 8px; font-size: 13px; text-align: center; color: var(--muted); }
.ui-asw-fd__foot { margin: 8px -18px 0; padding: 8px 18px 0; font-size: 12px; line-height: 16px; text-align: center; color: var(--faint); box-shadow: inset 0 0.5px 0 var(--line); }
/* Quick Look: a card over the list */
.ui-asw-ql { position: absolute; z-index: 5; top: 6px; left: -6px; right: -6px; display: flex; flex-direction: column; max-height: 380px; overflow: hidden; border-radius: calc(var(--aui-radius-lg, 10px) + 2px); background: var(--aui-overlay, var(--card)); box-shadow: var(--aui-shadow-popover, 0 0 0 0.5px var(--line-strong), 0 18px 44px -12px rgba(0, 0, 0, 0.26)); animation: ui-asw-ql 200ms var(--asw-out) both; }
.ui-asw-ql__head { display: flex; flex: none; align-items: center; gap: 10px; min-width: 0; height: 48px; padding: 0 8px 0 12px; box-shadow: inset 0 -0.5px 0 var(--line); }
.ui-asw-ql__head .ui-asw-art__title { flex: 1; }
.ui-asw-ql__close { display: grid; flex: none; place-items: center; width: 24px; height: 24px; border-radius: calc(var(--aui-radius-sm, 5px) + 1px); color: var(--muted); transition: background-color 120ms ease; }
.ui-asw-ql__close:hover { background: var(--hover); color: var(--ink); }
.ui-asw-ql__body { flex: 1; min-height: 0; overflow: auto; padding: 14px 14px 16px; }
.ui-asw-ql__body .ui-asw-wire { display: block; width: 100%; height: auto; }
.ui-asw-ql__none { margin: 0; padding: 28px 0; font-size: 13px; text-align: center; color: var(--muted); }
.ui-asw-ql__sheet { overflow-x: auto; }
.ui-asw-ql__sheet table { border-collapse: collapse; font-size: 12px; font-variant-numeric: tabular-nums; }
.ui-asw-ql__sheet td { height: 24px; padding: 0 8px; white-space: nowrap; color: var(--ink); box-shadow: inset -0.5px -0.5px 0 var(--asw-grid); }
.ui-asw-ql__sheet td[data-num] { text-align: right; }
.ui-asw-ql__sheet tr[data-head] td { font-weight: 500; }
.ui-asw-fd__rename { flex: 1; min-width: 0; height: 22px; margin: 0 -4px; padding: 0 4px; border: 0; border-radius: calc(var(--aui-radius-sm, 5px) - 1px); background: var(--card); box-shadow: inset 0 0 0 1px var(--line-strong); font: inherit; font-size: inherit; color: var(--ink); outline: none; }
.ui-asw-fd__tile .ui-asw-fd__rename { flex: none; width: 100%; margin: 0; text-align: center; font-size: 12px; }
@keyframes ui-asw-ql { from { opacity: 0; transform: scale(0.97) translateY(-4px); } }

/* ---------- Context menus – the shell's menu (.ui-as-menu), placed at the pointer ---------- */
.ui-asw-ctx { position: fixed; z-index: 80; min-width: 200px; max-width: min(300px, calc(100vw - 12px)); animation: ui-asw-ctx 120ms ease both; }
.ui-asw-ctx__row { position: relative; display: block; }
.ui-asw-ctx .ui-as-menu__item[aria-disabled] { opacity: 0.4; cursor: default; }
.ui-asw-ctx .ui-as-menu__item[aria-disabled]:is(:hover, :focus-visible) { background: none; }
.ui-asw-ctx .ui-as-menu__item:focus { outline: none; background: var(--hover); color: var(--ink); }
.ui-asw-ctx .ui-as-menu__item[data-open] { background: var(--hover); color: var(--ink); }
.ui-asw-ctx .ui-as-menu__item[aria-haspopup] > svg:last-child { color: var(--faint); }
.ui-asw-ctx .ui-as-menu__item[data-danger]:is(:hover, :focus) { background: color-mix(in srgb, var(--aui-red, #f04438) 8%, transparent); }
.ui-asw-ctx__sub { position: absolute; z-index: 1; top: -4px; left: calc(100% + 4px); min-width: 224px; }
.ui-asw-ctx__sub[data-flip] { right: calc(100% + 4px); left: auto; }
/* A bridge over the gap, so the pointer can cross to the submenu */
.ui-asw-ctx__sub::before { content: ""; position: absolute; top: 0; bottom: 0; left: -8px; width: 8px; }
.ui-asw-ctx__sub[data-flip]::before { right: -8px; left: auto; }
@keyframes ui-asw-ctx { from { opacity: 0; transform: scale(0.98); } }

@keyframes ui-asw-fade { from { opacity: 0; } }

@media (prefers-reduced-motion: reduce) {
  .ui-asw-ql, .ui-asw-tip, .ui-asw-ctx { animation: none !important; }
  .ui-asw-tool:active:not([aria-disabled]), .ui-asw-btn:active { transform: none; }
}

/* Touch: every control at least 36px */
@media (pointer: coarse) {
  .ui-asw-btn { min-height: 36px; }
  .ui-asw-tool, .ui-asw-fd__crumb { min-width: 36px; height: 36px; }
  .ui-asw-fd__row { height: 40px; }
}

`;

const workspaceCss = /* css */ `
.ui-asw { display: flex; flex-direction: column; gap: 22px; padding: 0 2px; }

/* Address field: a white pill with a hairline, like the search fields in the screens */
.ui-asw__address {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 34px;
  padding: 0 12px;
  border-radius: calc(var(--aui-radius, 7px) + 3px);
  background: var(--card);
  box-shadow: inset 0 0 0 0.5px var(--line-strong), var(--shadow);
  color: var(--faint);
  cursor: text;
}
.ui-asw__address:focus-within { box-shadow: inset 0 0 0 1px var(--line-strong), var(--shadow); }
.ui-asw__address input {
  flex: 1;
  min-width: 0;
  border: 0;
  outline: 0;
  background: transparent;
  font: inherit;
  font-size: 13px;
  color: var(--ink);
}
.ui-asw__address input::placeholder { color: var(--placeholder); }

.ui-asw__section { display: flex; flex-direction: column; gap: 8px; }
.ui-asw__label { margin: 0; padding: 0 4px; font-size: 12px; line-height: 16px; font-weight: 500; color: var(--muted); }

/* Rows sit on one white card, split by hairlines – the same card style as the screens */
.ui-asw__list {
  display: flex;
  flex-direction: column;
  padding: 4px;
  border-radius: calc(var(--aui-radius-lg, 10px) + 2px);
  background: var(--card);
  box-shadow: inset 0 0 0 0.5px var(--line-strong), var(--shadow);
}
.ui-asw__row { position: relative; display: flex; align-items: center; border-radius: calc(var(--aui-radius-sm, 5px) + 3px); transition: background-color 140ms ease; }
.ui-asw__row + .ui-asw__row::before {
  content: "";
  position: absolute;
  top: -0.5px;
  left: 44px;
  right: 8px;
  height: 1px;
  background: var(--line-soft);
}
.ui-asw__open {
  display: flex;
  flex: 1;
  min-width: 0;
  align-items: center;
  gap: 10px;
  height: 40px;
  padding: 0 8px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 3px);
  text-align: left;
  transition: background-color 140ms ease;
}
/* The hover fills the whole row, the ⋯ button at its end included */
.ui-asw__row:hover { background: var(--hover); }
.ui-asw__source { height: 48px; flex: none; }

.ui-asw__tile {
  display: grid;
  flex: none;
  place-items: center;
  width: 26px;
  height: 26px;
  border-radius: 7px;
  background: var(--chip);
  color: var(--text);
}
.ui-asw__tile--mark { font-size: 11.5px; font-weight: 500; color: var(--muted); }

.ui-asw__text { display: flex; flex: 1; min-width: 0; flex-direction: column; }
.ui-asw__name { overflow: hidden; font-size: 13px; line-height: 18px; font-weight: 450; color: var(--ink); white-space: nowrap; text-overflow: ellipsis; }
.ui-asw__open > .ui-asw__name { flex: 1; }
.ui-asw__meta { overflow: hidden; font-size: 12px; line-height: 16px; color: var(--muted); white-space: nowrap; text-overflow: ellipsis; }
.ui-asw__empty { margin: 0; padding: 0 4px; font-size: 13px; line-height: 18px; color: var(--faint); }

.ui-asw__kbd {
  flex: none;
  padding: 1px 6px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 1px);
  background: var(--chip);
  font: inherit;
  font-size: 11px;
  line-height: 18px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  transition: opacity 140ms ease;
}
/* The shortcut sits flush right; on hover the ⋯ button takes its place */
.ui-asw__more {
  position: absolute;
  top: 50%;
  right: 6px;
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 2px);
  color: var(--faint);
  opacity: 0;
  translate: 0 -50%;
  transition: opacity 140ms ease, background-color 140ms ease;
}
.ui-asw__row:is(:hover, :focus-within) .ui-asw__more { opacity: 1; }
.ui-asw__row:is(:hover, :focus-within) .ui-asw__kbd { opacity: 0; }
.ui-asw__more:hover { background: var(--hover); color: var(--text); }
.ui-asw__go { flex: none; color: var(--faint); opacity: 0; translate: -2px 0; transition: opacity 140ms ease, translate 200ms var(--ease); }
.ui-asw__source:hover .ui-asw__go { opacity: 1; translate: 0 0; }
`;

const css = /* css */ `
/* Design tokens – same values as the affectUI library components */
.ui-as {
  --tray: #f4f4f4;
  --card: var(--aui-surface, #ffffff);
  /* Like the affectUI library: gray page, the main area is a rounded panel on it */
  --page: var(--aui-subtle, #f8f8f8);
  /* The double sidebar's chats: between the page and the panel */
  --side-tint: var(--aui-subtle, #fcfcfc);
  --panel: var(--aui-surface, #ffffff);
  --radius-panel: calc(var(--aui-radius-lg, 10px) + 2px);
  --frame-gap: 8px;
  /*
   * One head row for the whole shell: every screen's top bar, the work area's tabs and the header across the app are
   * this tall, and the corner icons (sidebar, work area) sit on its centerline. The heads start under the main panel's
   * top border (--as-head-edge, 0 in the flat layout), so the line is frame gap + edge + half the head from the top.
   */
  --as-head-h: 48px;
  --as-head-edge: 1px;
  /* Clean: hairlines drawn with shadows, text in stepped opacities of black, Inter in 400 and 500 */
  --line: var(--aui-border, rgba(0, 0, 0, 0.06));
  --line-soft: var(--aui-border, rgba(0, 0, 0, 0.04));
  --strong: var(--aui-ink, #000000);
  --ink: var(--aui-ink, rgba(0, 0, 0, 0.88));
  --text: var(--aui-muted, rgba(0, 0, 0, 0.61));
  --muted: var(--aui-muted, rgba(0, 0, 0, 0.45));
  --faint: var(--aui-placeholder, rgba(0, 0, 0, 0.27));
  --placeholder: var(--aui-placeholder, rgba(0, 0, 0, 0.27));
  --chip: var(--aui-subtle, rgba(0, 0, 0, 0.04));
  --chip-hover: rgba(0, 0, 0, 0.06);
  --hover: rgba(0, 0, 0, 0.03);
  --accent: var(--aui-info, #0070d6);
  --ui-as-focus: var(--aui-focus, rgba(0, 96, 226, 0.7));
  --button: var(--aui-primary, rgba(0, 0, 0, 0.88));
  --button-hover: rgba(0, 0, 0, 0.78);
  --accent-soft: rgba(0, 96, 226, 0.08);
  --shadow: 0 1px 1px -1px rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  /* A card: a 0.5px hairline and a soft drop; the frame around previews is a faint gray with the same hairline */
  --card-shadow: var(--aui-shadow-card, 0 0 0 0.5px var(--aui-border, rgba(0, 0, 0, 0.06)), 0 1px 1px -1px rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.05));
  --card-shadow-hover: 0 0 0 0.5px var(--aui-border-strong, rgba(0, 0, 0, 0.1)), 0 2px 6px -1px rgba(0, 0, 0, 0.08), 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --frame: rgba(0, 0, 0, 0.02);
  --line-strong: var(--aui-border-strong, rgba(0, 0, 0, 0.1));
  /* An edge you can drag, while the pointer is on it */
  --edge-active: var(--aui-border-strong, rgba(0, 0, 0, 0.14));
  --subtle: var(--aui-subtle, #fafafa);
  --disabled: #dcdcdc;
  --on-ink: #ffffff;
  --ink-hover: #3a3a3a;
  --select: rgba(0, 0, 0, 0.055);
  --toggle-off: #e2e2e2;
  --switch-on: #171717;
  --logo-bg: #ffffff;
  --ease: var(--aui-ease, cubic-bezier(0.22, 1, 0.36, 1));
  --font: var(--aui-font, "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif);
}

/* Library components inside a screen are their clean versions – same scale and hairlines as the shell; approval
   actions stay at their natural width, at the end of the row */
.ui-as .ui-apc__foot .ui-apc__actions { flex: 0 0 auto; margin-left: auto; }

/* Dark mode: follows a .dark class or data-theme="dark" on any ancestor, e.g. <html>.
   Every screen draws from these tokens, and the library parts inside switch on their own. */
:where(.dark, [data-theme="dark"]) .ui-as {
  --tray: #1d1d20;
  --card: var(--aui-surface, #1b1b1e);
  --page: var(--aui-subtle, #0e0e10);
  --side-tint: var(--aui-subtle, #111113);
  --panel: var(--aui-surface, #141416);
  --line: var(--aui-border, rgba(255, 255, 255, 0.08));
  --line-soft: var(--aui-border, rgba(255, 255, 255, 0.05));
  --strong: var(--aui-ink, #ffffff);
  --ink: var(--aui-ink, rgba(255, 255, 255, 0.9));
  --text: var(--aui-muted, rgba(255, 255, 255, 0.62));
  --muted: var(--aui-muted, rgba(255, 255, 255, 0.45));
  --faint: var(--aui-placeholder, rgba(255, 255, 255, 0.28));
  --placeholder: var(--aui-placeholder, rgba(255, 255, 255, 0.28));
  --chip: var(--aui-subtle, rgba(255, 255, 255, 0.06));
  --chip-hover: rgba(255, 255, 255, 0.09);
  --hover: rgba(255, 255, 255, 0.05);
  --accent: var(--aui-info, #4fb0fe);
  --ui-as-focus: var(--aui-focus, rgba(77, 141, 255, 0.8));
  --button: var(--aui-primary, #f4f4f5);
  --button-hover: #ffffff;
  --accent-soft: color-mix(in srgb, var(--aui-blue, #4fb0fe) 16%, transparent);
  --shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
  --card-shadow: var(--aui-shadow-card, 0 0 0 0.5px var(--aui-border, rgba(255, 255, 255, 0.1)), 0 1px 2px rgba(0, 0, 0, 0.4));
  --card-shadow-hover: 0 0 0 0.5px var(--aui-border-strong, rgba(255, 255, 255, 0.16)), 0 2px 6px rgba(0, 0, 0, 0.4);
  --frame: rgba(255, 255, 255, 0.03);
  --line-strong: var(--aui-border-strong, rgba(255, 255, 255, 0.14));
  --edge-active: var(--aui-border-strong, rgba(255, 255, 255, 0.18));
  --subtle: var(--aui-subtle, #18181b);
  --disabled: #3a3a3f;
  --on-ink: #111113;
  --ink-hover: #d4d4d8;
  --select: rgba(255, 255, 255, 0.08);
  --toggle-off: #3a3a3f;
  --switch-on: var(--aui-blue, #4fb0fe);
  --logo-bg: #2a2a2e;
  color-scheme: dark;
}

.ui-as, .ui-as *, .ui-as *::before, .ui-as *::after { box-sizing: border-box; }
.ui-as {
  font-family: var(--font);
  font-size: 13px;
  color: var(--text);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
:where(.ui-as) button {
  appearance: none;
  margin: 0;
  padding: 0;
  border: 0;
  background: none;
  font: inherit;
  color: inherit;
  cursor: pointer;
}
:where(.ui-as) svg { display: block; flex: none; }
/* The shell's ring at the lowest weight that still beats a host page's bare :focus-visible – any screen rule
   (.ui-x:focus-visible) wins by design, not by the order of the style tags */
.ui-as :where(:focus-visible) { outline: 2px solid var(--ui-as-focus); outline-offset: 2px; }
.ui-as textarea:focus-visible, .ui-as input:focus-visible { outline: none; }

/* App frame shared by all screens: sidebar on the gray page, main area as a rounded white panel */

.ui-as-app {
  /* The layout answers to the app's own width (container queries), not the window – see the end of this file */
  container: ui-as / inline-size;
  position: relative;
  display: grid;
  /* Sidebar · main area · work area on the right (0 when there is none or it's closed); widths come from the drag handles */
  --side-col: var(--side-w, 272px);
  --panel-col: 0px;
  grid-template-columns: var(--side-col) minmax(0, 1fr) var(--panel-col);
  height: 100%;
  min-height: 560px;
  background: var(--page);
  transition: grid-template-columns 420ms var(--ease);
}
.ui-as-app[data-panel="open"] { --panel-col: var(--panel-w, 360px); }
/* Dragging an edge: the columns follow the pointer at once */
.ui-as-app[data-resizing] { transition: none; cursor: col-resize; user-select: none; }
.ui-as-app[data-resizing] :is(.ui-as-main, .ui-as-side, .ui-as-panel) { transition: none; }

/* ---------- Sidebar ---------- */

.ui-as-side {
  display: flex;
  flex-direction: column;
  width: var(--side-w, 272px);
  min-height: 0;
  padding: 10px 8px 8px 10px;
  transition: opacity 260ms ease, translate 420ms var(--ease), visibility 0s;
}
/* The rail: the sidebar keeps its layout and is only cut to the icons – every row stays where it was,
   at its height, with its own fill. Words go; each row names itself on hover */
.ui-as-app[data-collapsed] { --side-col: 48px; }
.ui-as-app[data-collapsed] > .ui-as-side { width: 100%; overflow: hidden; }
.ui-as-app[data-collapsed] > .ui-as-side > * { flex-shrink: 0; width: calc(var(--side-w, 272px) - 20px); }
.ui-as-app[data-collapsed] :is(.ui-as-side__name, .ui-as-side__workspace > svg, .ui-as-side__text, .ui-as-side__count, .ui-as-user__name, .ui-as-user > svg, .ui-as-side__search > span, .ui-as-side__search > kbd) { display: none; }
.ui-as-app[data-collapsed] :is(.ui-as-side__item, .ui-as-side__search, .ui-as-side__workspace, .ui-as-user-menu > .ui-as-user) { flex: none; width: fit-content; }
.ui-as-app[data-collapsed] .ui-as-side__search { padding-right: 8px; }
/* Group titles keep their room; a short hairline stands in for the words */
.ui-as-app[data-collapsed] .ui-as-side__label { color: transparent; white-space: nowrap; overflow: hidden; background: linear-gradient(var(--line-strong), var(--line-strong)) 0 50% / 28px 0.5px no-repeat; }
.ui-as-app[data-collapsed] .ui-as-side__empty { visibility: hidden; }
.ui-as-app[data-collapsed] .ui-as-side__foot > :not(.ui-as-user-menu) { display: none; }
/* The toggle lies over the workspace logo and takes its place while the pointer is on the sidebar; a settings sidebar drops its way back and shows the toggle */
.ui-as-app[data-collapsed] :is(.ui-as-side__head, .ui-as-bar--side) { position: relative; }
.ui-as-app[data-collapsed] .ui-as-side__tools,
.ui-as-app[data-collapsed] > .ui-as-bar--side > .ui-as-icon-btn { position: absolute; top: 0; left: 0; opacity: 0; transition: opacity 140ms ease; }
.ui-as-app[data-collapsed] > .ui-as-bar--side > .ui-as-icon-btn { top: 10px; left: 10px; margin: 0; }
.ui-as-app[data-collapsed] :is(.ui-as-side__tools, .ui-as-bar--side) .ui-as-icon-btn { width: 30px; }
.ui-as-app[data-collapsed]:has(> :is(.ui-as-side, .ui-as-bar--side):hover, :is(.ui-as-side__head, .ui-as-bar--side) :focus-visible) :is(.ui-as-side__tools, .ui-as-bar--side > .ui-as-icon-btn),
.ui-as-app[data-collapsed] :is(.ui-as-side__head, .ui-as-bar--side):has(> .ui-as-side__back) :is(.ui-as-side__tools, .ui-as-icon-btn) { opacity: 1; }
.ui-as-app[data-collapsed]:has(> :is(.ui-as-side, .ui-as-bar--side):hover, :is(.ui-as-side__head, .ui-as-bar--side) :focus-visible) .ui-as-side__logo { opacity: 0; }
.ui-as-app[data-collapsed] .ui-as-side__logo { transition: opacity 140ms ease; }
.ui-as-app[data-collapsed] .ui-as-side__back { visibility: hidden; }

/* The icon to bring the sidebar back, in the top left corner of the main area */
.ui-as-side-open {
  position: fixed;
  z-index: 20;
  /* On the head row's centerline – the button is 28px tall */
  top: calc(var(--frame-gap) + var(--as-head-edge) + (var(--as-head-h) - 28px) / 2);
  left: calc(var(--frame-gap) + 9px);
  visibility: hidden;
  opacity: 0;
  transition: opacity 200ms ease, visibility 0s 200ms;
}


.ui-as-side__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 28px;
}
/* The workspace's name takes what the tools leave and shortens with … – the full name on hover */
.ui-as-side__head { gap: 4px; }
.ui-as-side__head > .ui-as-anchor { display: flex; min-width: 0; }
.ui-as-side__workspace {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  max-width: 100%;
  text-align: left;
  height: 30px;
  padding: 0 6px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 3px);
  font-size: 14px;
  font-weight: 500;
  color: var(--ink);
}
.ui-as-side__workspace svg { color: var(--faint); }
.ui-as-side__name { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.ui-as-side__tools { flex: none; }
.ui-as-side__workspace:hover { background: var(--hover); }
/* Settings sidebar: "‹ Settings" in place of the workspace */
.ui-as-side__back { display: flex; align-items: center; gap: 6px; min-width: 0; height: 30px; padding: 0 8px 0 4px; border-radius: calc(var(--aui-radius-sm, 5px) + 3px); font-size: 14px; font-weight: 500; color: var(--strong); }
.ui-as-side__back svg { color: var(--muted); transition: translate 200ms var(--ease); }
.ui-as-side__back:hover { background: var(--hover); }
.ui-as-side__back:hover svg { translate: -2px 0; }
.ui-as-user-menu { display: flex; }
.ui-as-user-menu > .ui-as-user { flex: 1; min-width: 0; }
.ui-as-user-menu .ui-as-menu { width: 100%; }
/* The account menu's head: picture, name and email */
.ui-as-account { display: flex; align-items: center; gap: 10px; padding: 8px 10px 10px; margin-bottom: 4px; border-bottom: 1px solid var(--line); }
.ui-as-account img, .ui-as-account .ui-as-user__initials { flex: none; width: 30px; height: 30px; border-radius: 50%; font-size: 11px; }
.ui-as-account__text { display: grid; min-width: 0; }
.ui-as-account__text b { overflow: hidden; font-size: 13px; font-weight: 500; line-height: 18px; white-space: nowrap; text-overflow: ellipsis; color: var(--ink); }
.ui-as-account__text small { overflow: hidden; font-size: 12px; line-height: 16px; white-space: nowrap; text-overflow: ellipsis; color: var(--muted); }
/* The workspace menu runs as wide as the sidebar */
.ui-as-ws-menu .ui-as-menu { width: 240px; }
/* Workspace logo on a white tile with a small, tight shadow below */
.ui-as-side__logo {
  display: grid;
  place-items: center;
  width: 18px;
  height: 18px;
  margin-right: 1px;
  border-radius: 5px;
  background: var(--logo-bg);
  box-shadow: 0 0 0 0.5px rgba(0, 0, 0, 0.08), 0 1px 1px rgba(0, 0, 0, 0.12);
}
.ui-as-side__logo svg { width: 11px; height: auto; }
.ui-as-side__tools { display: flex; gap: 2px; }

.ui-as-icon-btn {
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 3px);
  color: var(--muted);
  transition: background-color 140ms ease, color 140ms ease;
}
.ui-as-icon-btn:hover { background: var(--hover); color: var(--ink); }

.ui-as-side__nav {
  display: flex;
  flex: 0 1 auto;
  flex-direction: column;
  min-height: 0;
  margin-top: 12px;
  /* A long list scrolls under the head, the cards at the bottom stay */
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-width: none;
}
.ui-as-side__nav::-webkit-scrollbar { display: none; }
/* Rows keep their height – the list scrolls instead of squeezing them */
.ui-as-side__nav > *, .ui-as-side__nav .ui-as-side__group > * { flex-shrink: 0; }
.ui-as-side__foot { flex: none; }

/* Chats: no icon – loose ones start at the rows' left edge, the ones in a folder line up with its name;
   ⋯ and the pin show on hover */
.ui-as-side { position: relative; }
.ui-as-chats, .ui-as-chat-pinned { margin-top: 6px; }
.ui-as-chats-skel .ui-as-side__label { display: flex; align-items: center; height: 16px; }
.ui-as-chat-folder-row { position: relative; display: flex; flex: none; }
.ui-as-chat-folder-row:is(:hover, :has(:focus-visible), [data-menu]) > .ui-as-chat-folder { padding-right: 56px; background: var(--hover); color: var(--ink); }
.ui-as-chat-folder-row:is(:hover, :has(:focus-visible), [data-menu]) .ui-as-chat-row__tools { opacity: 1; }
.ui-as-chat-folder__rename { padding-left: 29px; }
.ui-as-chat-folder-row[draggable="true"] { cursor: grab; }
.ui-as-chat-folder > .ui-as-side__text { flex: 0 1 auto; }
.ui-as-chat-folder__arrow { display: grid; flex: none; place-items: center; margin-left: -4px; color: var(--faint); opacity: 0; transition: opacity 120ms ease, transform 200ms var(--ease); }
.ui-as-chat-folder[aria-expanded="true"] .ui-as-chat-folder__arrow { transform: rotate(90deg); }
.ui-as-chat-folder-row:is(:hover, :has(:focus-visible)) .ui-as-chat-folder__arrow { opacity: 1; }
.ui-as-chat-folder-row[data-dragging] { opacity: 0.4; }
.ui-as-chat-folder-row[data-drop]::after { content: ""; position: absolute; left: 6px; right: 6px; height: 2px; border-radius: 1px; background: var(--accent); pointer-events: none; }
.ui-as-chat-folder-row[data-drop="before"]::after { top: -1px; }
.ui-as-chat-folder-row[data-drop="after"]::after { bottom: -1px; }
.ui-as-chat-row { position: relative; display: flex; flex: none; }
.ui-as-chat-row > .ui-as-chat-row__link { padding-left: 6px; }
.ui-as-chat-row[data-indent] > .ui-as-chat-row__link { padding-left: 30px; }
.ui-as-chat-folder__chats { display: flex; flex-direction: column; }
.ui-as-chat-folder__empty { padding-left: 30px; }
/* A chat dragged over a folder: the folder lights up */
.ui-as-chat-folder[data-drop] { background: var(--hover); box-shadow: inset 0 0 0 1px var(--accent); color: var(--ink); }
.ui-as-chat-row__link .ui-as-side__text { flex: 1; }
.ui-as-chat-row:is(:hover, :has(:focus-visible), [data-menu]) .ui-as-chat-row__link { padding-right: 56px; background: var(--hover); color: var(--ink); }
/* The open chat: only the soft hover fill behind it, no heavier type */
.ui-as-chat-row > .ui-as-chat-row__link[aria-current="page"] { background: var(--hover); font-weight: 400; color: var(--ink); }
.ui-as-chat-row__dot { flex: none; width: 6px; height: 6px; margin-left: 6px; border-radius: 50%; background: var(--accent); }
.ui-as-chat-row__tools { position: absolute; top: 3px; right: 4px; display: flex; gap: 2px; opacity: 0; transition: opacity 120ms ease; }
.ui-as-chat-row:is(:hover, :has(:focus-visible), [data-menu]) .ui-as-chat-row__tools { opacity: 1; }
.ui-as-chat-row__tool { display: grid; place-items: center; width: 24px; height: 24px; border-radius: calc(var(--aui-radius-sm, 5px) + 1px); color: var(--muted); transition: background-color 120ms ease, color 120ms ease; }
.ui-as-chat-row__tool:hover, .ui-as-chat-row__tool[aria-expanded="true"] { background: var(--chip); color: var(--ink); }
.ui-as-chat-row__rename { width: 100%; height: 30px; padding: 0 8px 0 5px; border: 0; border-radius: calc(var(--aui-radius-sm, 5px) + 3px); outline: 0; background: var(--card); box-shadow: inset 0 0 0 1px var(--line-strong); font: inherit; font-size: 13px; color: var(--ink); }
/* Dragging: the row fades, a line shows where it lands */
.ui-as-chat-row[data-indent] .ui-as-chat-row__rename { padding-left: 29px; }
.ui-as-chat-row[draggable="true"] { cursor: grab; }
.ui-as-chat-row[data-dragging] { opacity: 0.4; }
.ui-as-chat-row[data-drop]::after { content: ""; position: absolute; left: 6px; right: 6px; height: 2px; border-radius: 1px; background: var(--accent); pointer-events: none; }
.ui-as-chat-row[data-drop="before"]::after { top: -1px; }
.ui-as-chat-row[data-drop="after"]::after { bottom: -1px; }
.ui-as-chat-menu { position: absolute; z-index: 30; width: 232px; animation: ui-as-popover-up 160ms var(--ease) both; }
.ui-as-chat-menu .ui-as-menu__subpanel { min-width: 180px; }
/* The rail has no room for names */
.ui-as-app[data-collapsed] :is(.ui-as-chats, .ui-as-chat-folder__chats, .ui-as-chat-row, .ui-as-chat-row__tools) { display: none; }
.ui-as-app[data-collapsed] .ui-as-chat-pinned:not(:has(.ui-as-chat-folder)) { display: none; }
.ui-as-app[data-collapsed] .ui-as-chat-folder__arrow { display: none; }
/* Search opens the Command Menu over the app, near the top, on a soft scrim */
.ui-as-cmd {
  position: absolute;
  inset: 0;
  z-index: 60;
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: min(12vh, 96px) 16px 16px;
  background: rgba(0, 0, 0, 0.12);
  animation: ui-as-cmd-fade 200ms ease both;
}
.ui-as-cmd > * { width: 100%; max-width: 560px; animation: ui-as-cmd-in 260ms var(--ease) both; }
@keyframes ui-as-cmd-fade { from { opacity: 0; } }
@keyframes ui-as-cmd-in { from { opacity: 0; transform: translateY(-6px) scale(0.98); } }
:where(.dark, [data-theme="dark"]) .ui-as-cmd { background: rgba(0, 0, 0, 0.45); }

/* The search field under the workspace */
.ui-as-side__search {
  display: flex;
  flex: none;
  align-items: center;
  gap: 8px;
  width: 100%;
  height: 30px;
  margin-top: 12px;
  padding: 0 6px 0 8px;
  border-radius: calc(var(--aui-radius, 7px) + 1px);
  background: var(--card);
  box-shadow: var(--card-shadow);
  font-size: 13px;
  color: var(--muted);
  text-align: left;
  transition: box-shadow 140ms ease, color 140ms ease;
}
.ui-as-side__search svg { color: var(--faint); }
.ui-as-side__search:hover { box-shadow: var(--card-shadow-hover); color: var(--text); }
.ui-as-side__search kbd { margin-left: auto; padding: 0 5px; border-radius: calc(var(--aui-radius-sm, 5px) - 1px); box-shadow: inset 0 0 0 0.5px var(--line-strong); font: inherit; font-size: 11px; line-height: 18px; color: var(--faint); }
.ui-as-side__group { display: flex; flex-direction: column; }
.ui-as-side__group + .ui-as-side__group { margin-top: 10px; }
.ui-as-side__group + .ui-as-side__group > .ui-as-side__label:first-child { margin-top: 6px; }
.ui-as-side__count { flex: none; margin-left: auto; font-size: 12px; color: var(--faint); font-variant-numeric: tabular-nums; }
.ui-as-side__empty { padding: 4px 6px; font-size: 13px; color: var(--faint); }
.ui-as-side__item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  height: 30px;
  padding: 0 6px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 3px);
  font-size: 13px;
  color: var(--text);
  text-align: left;
  transition: background-color 140ms ease, color 140ms ease;
}
.ui-as-side__item svg { flex: none; color: var(--muted); }
/* One line per row – long names shorten with …, the full name on hover */
.ui-as-side__text { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.ui-as-side__item:hover { background: var(--hover); color: var(--ink); }
.ui-as-side__item[aria-current="page"] { background: var(--select); color: var(--ink); font-weight: 500; }
.ui-as-side__item[aria-current="page"] svg { color: var(--ink); }
a.ui-as-side__item { text-decoration: none; cursor: pointer; }
.ui-as-side__label {
  margin: 16px 0 4px;
  padding: 0 6px;
  font-size: 12.5px;
  line-height: 16px;
  color: var(--faint);
}

.ui-as-side__foot {
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: auto;
}

/* "What's new" – two cards stacked, the second one peeks out above */
.ui-as-news { position: relative; padding-top: 30px; }
.ui-as-news__card {
  position: relative;
  z-index: 1;
  padding: 10px 10px 0;
  border-radius: calc(var(--aui-radius-lg, 10px) + 2px);
  background: var(--card);
  box-shadow: var(--card-shadow);
}
.ui-as-news__card--back {
  position: absolute;
  z-index: 0;
  top: 0;
  left: 10px;
  right: 10px;
  height: 48px;
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 9px 9px 0;
  background: var(--card);
  color: var(--faint);
}
.ui-as-news[data-count="1"] { padding-top: 0; }
.ui-as-news[data-count="1"] .ui-as-news__card--back { display: none; }
.ui-as-news__top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.ui-as-news__kicker { min-width: 0; overflow: hidden; font-size: 11.5px; line-height: 15px; font-weight: 500; color: var(--accent); white-space: nowrap; text-overflow: ellipsis; }
.ui-as-news__card--back svg, .ui-as-news__close { flex: none; }
.ui-as-news__close {
  display: grid;
  place-items: center;
  width: 18px;
  height: 18px;
  margin: -2px -2px 0 0;
  border-radius: calc(var(--aui-radius-sm, 5px) + 1px);
  color: var(--faint);
}
.ui-as-news__close:hover { background: var(--hover); color: var(--ink); }
.ui-as-news__title { display: block; margin-top: 4px; font-size: 13px; line-height: 18px; font-weight: 500; color: var(--ink); }
.ui-as-news__more {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: calc(100% + 20px);
  height: 34px;
  margin: 10px -10px 0;
  padding: 0 10px;
  border-top: 1px solid var(--line-soft);
  border-radius: 0 0 calc(var(--aui-radius-lg, 10px) + 2px) calc(var(--aui-radius-lg, 10px) + 2px);
  font-size: 12.5px;
  color: var(--muted);
}
.ui-as-news__more:hover { background: var(--subtle); color: var(--ink); }

/* Onboarding progress */
/* Getting started: one card that morphs between the short summary and the full Setup Guide */
.ui-as-guide-slot { position: relative; flex: none; }
.ui-as-guide {
  position: absolute;
  z-index: 6;
  right: 0;
  bottom: 0;
  left: 0;
  overflow: hidden;
  border-radius: calc(var(--aui-radius-lg, 10px) + 2px);
  background: var(--card);
  box-shadow: var(--card-shadow);
  transition: height 480ms var(--ease), border-radius 480ms var(--ease), box-shadow 200ms ease;
}
.ui-as-guide:not([data-open]):hover { box-shadow: 0 0 0 1px var(--line-strong), 0 2px 6px rgba(0, 0, 0, 0.05); }
.ui-as-guide[data-open] { border-radius: calc(var(--aui-radius-lg, 10px) + 2px); box-shadow: 0 0 0 0.5px var(--line-strong), 0 8px 24px -10px rgba(0, 0, 0, 0.14); }
.ui-as-guide__content { animation: ui-as-guide-in 360ms var(--ease) both; }
/* Closing: the guide fades out while the card shrinks back to the summary's height */
.ui-as-guide[data-state="closing"] .ui-as-guide__content { animation: ui-as-guide-out 260ms ease both; }
.ui-as-guide[data-state="closing"] { border-radius: calc(var(--aui-radius-lg, 10px) + 2px); box-shadow: var(--card-shadow); }
@keyframes ui-as-guide-out { to { opacity: 0; } }
@keyframes ui-as-guide-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
.ui-as-steps {
  display: grid;
  gap: 3px;
  width: 100%;
  padding: 10px 10px 9px;
  text-align: left;
}
/* The library's Setup Guide inside the card: its own frame drops away, the card is the frame */
/* Opening and closing belong to the card, so the guide's own minimize button is left out */
.ui-as-guide .ui-sgdc__icon-btn[aria-expanded] { display: none; }
.ui-as-guide .ui-sgdc { zoom: 0.86; max-width: none; padding: 12px 4px 4px; border: 0; border-radius: 0; background: transparent; box-shadow: none; }
.ui-as-steps__title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 13px;
  line-height: 18px;
  font-weight: 500;
  color: var(--ink);
}
.ui-as-steps__title svg { color: var(--faint); }
.ui-as-steps__sub { font-size: 12.5px; line-height: 17px; color: var(--muted); }
.ui-as-steps__cheer { color: var(--accent); }
.ui-as-steps__bar {
  height: 4px;
  margin-top: 6px;
  overflow: hidden;
  border-radius: 2px;
  background: var(--tray);
}
.ui-as-steps__bar i { display: block; min-width: 6px; height: 100%; border-radius: inherit; background: var(--accent); transition: width 600ms var(--ease); }

.ui-as-user {
  display: flex;
  align-items: center;
  gap: 7px;
  height: 34px;
  padding: 0 6px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 3px);
  font-size: 13px;
  font-weight: 500;
  color: var(--ink);
}
.ui-as-user__name { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.ui-as-user > :is(img, .ui-as-user__initials) { flex: none; }
.ui-as-user img { width: 21px; height: 21px; border-radius: 50%; object-fit: cover; box-shadow: 0 0 0 1px rgba(0, 0, 0, 0.06); }
.ui-as-user svg { margin-left: auto; color: var(--faint); }
.ui-as-user:hover { background: var(--hover); }

/* ---------- Main ---------- */

/* Main area as a rounded panel, like the affectUI library */
.ui-as-main {
  /* Screens reflow their content by the width of this panel */
  container: ui-as-main / inline-size;
  position: relative;
  min-width: 0;
  margin: var(--frame-gap) var(--frame-gap) var(--frame-gap) 0;
  overflow-y: auto;
  /* No rubber-band bounce at the ends – sticky headers stay put */
  overscroll-behavior: none;
  transition: margin-left 420ms var(--ease);
  border: 1px solid var(--line);
  border-radius: var(--radius-panel);
  background: var(--panel);
}


/* Shared page parts: headings, toolbars, tabs, search, empty states – affectUI tokens */

.ui-as-page {
  padding: 46px 32px 56px;
}
.ui-as-page-title {
  margin: 0;
  font-size: 24px;
  line-height: 32px;
  font-weight: 500;
  letter-spacing: -0.24px;
  color: var(--strong);
}
.ui-as-page-sub {
  margin: 4px 0 0;
  font-size: 13px;
  line-height: 20px;
  color: var(--text);
}
.ui-as-page-sub a { color: var(--text); text-decoration: underline; text-decoration-color: var(--line-strong); text-underline-offset: 3px; }
.ui-as-page-sub a:hover { color: var(--ink); text-decoration-color: currentColor; }

.ui-as-section-head { margin-top: 36px; }
/* Without tabs the tools sit on the same line as the title */
.ui-as-section-head[data-inline] { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
.ui-as-section-head[data-inline] .ui-as-toolbar { margin-top: 0; }
/* A section's title: a small uppercase label */
.ui-as-section-head__title {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0;
  font-size: 11px;
  line-height: 16px;
  font-weight: 500;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--muted);
}
.ui-as-section-head__count { font-variant-numeric: tabular-nums; color: var(--faint); }
.ui-as-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-top: 12px;
}
.ui-as-toolbar__left, .ui-as-toolbar__right { display: flex; align-items: center; gap: 6px; }

/* Tabs: small outlined buttons, the active one gets a soft gray fill */
.ui-as-tabs { position: relative; display: flex; gap: 8px; }
/* The gray pill glides under the active tab */
.ui-as-tabs__pill {
  position: absolute;
  top: 0;
  left: 0;
  height: 28px;
  border-radius: calc(var(--aui-radius, 7px) - 1px);
  background: var(--select);
  transition: transform 340ms var(--ease), width 340ms var(--ease);
}
.ui-as-tabs__tab {
  position: relative;
  height: 28px;
  padding: 0 10px;
  border-radius: calc(var(--aui-radius, 7px) - 1px);
  font-size: 13px;
  color: var(--text);
  box-shadow: inset 0 0 0 0.5px var(--line);
  transition: background-color 140ms ease, color 140ms ease;
}
.ui-as-tabs__tab:hover { color: var(--ink); }
.ui-as-tabs__tab[aria-selected="true"] { color: var(--ink); }

/* Small outlined tools */
.ui-as-tool {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  min-width: 28px;
  height: 28px;
  border-radius: calc(var(--aui-radius, 7px) - 1px);
  background: var(--card);
  box-shadow: var(--card-shadow);
  color: var(--text);
  transition: box-shadow 140ms ease, color 140ms ease;
}
.ui-as-tool svg { color: var(--muted); }
.ui-as-tool:hover { box-shadow: var(--card-shadow-hover); color: var(--ink); }
.ui-as-tool--label { padding: 0 10px; font-size: 13px; }

.ui-as-view {
  position: relative;
  display: flex;
  gap: 2px;
  height: 28px;
  padding: 2px;
  border-radius: calc(var(--aui-radius, 7px) - 1px);
  background: var(--chip);
}
.ui-as-view__btn {
  display: grid;
  place-items: center;
  width: 28px;
  border-radius: calc(var(--aui-radius, 7px) - 3px);
  color: var(--faint);
}
.ui-as-view__btn:hover { color: var(--ink); }
.ui-as-view__btn { position: relative; transition: color 160ms ease; }
.ui-as-view__btn[aria-checked="true"] { color: var(--ink); }
/* White pill that slides between grid and list */
.ui-as-view__pill {
  position: absolute;
  top: 2px;
  bottom: 2px;
  left: 2px;
  width: 28px;
  border-radius: calc(var(--aui-radius, 7px) - 3px);
  background: var(--card);
  box-shadow: var(--card-shadow);
  transition: transform 320ms var(--ease);
}
.ui-as-view[data-view="list"] .ui-as-view__pill { transform: translateX(30px); }

/* Search in a toolbar: the library's Text Field at the toolbar's width */
.ui-as-search { display: block; min-width: 0; }

/* Sort: the arrows turn half a circle for Z–A */
.ui-as-sort svg { transition: transform 360ms var(--ease); }
.ui-as-sort[data-dir="desc"] svg { transform: rotate(180deg); }
.ui-as-tool[data-active] { box-shadow: var(--card-shadow-hover); color: var(--ink); }
.ui-as-tool__count {
  display: grid;
  place-items: center;
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  border-radius: 999px;
  background: var(--ink);
  font-size: 11px;
  font-weight: 500;
  color: var(--on-ink);
  animation: ui-as-pop 220ms var(--ease);
}

/* Popovers: attached to their trigger, they fade and scale in from that side */
.ui-as-anchor { position: relative; display: inline-flex; }
/* The Filter Menu at its own size – 13px rows like every other menu, no shrinking */
.ui-as-popover--filter .ui-fmc { width: 300px; }
.ui-as-popover--filter .ui-fmc__title { font-size: 13px; line-height: 20px; }

.ui-as-menu {
  min-width: 168px;
  padding: 4px;
  border-radius: calc(var(--aui-radius-lg, 10px) + 2px);
  background: var(--aui-overlay, var(--card));
  box-shadow: var(--aui-shadow-popover, 0 0 0 0.5px var(--line-strong), 0 12px 32px -8px rgba(0, 0, 0, 0.16));
}
.ui-as-menu__title { display: block; padding: 6px 10px 4px; font-size: 12px; font-weight: 500; color: var(--faint); }
.ui-as-menu__list { display: grid; gap: 1px; }
.ui-as-menu__item {
  display: flex;
  align-items: center;
  gap: 9px;
  width: 100%;
  min-height: 32px;
  padding: 0 10px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 3px);
  font-size: 13px;
  color: var(--text);
  text-align: left;
  white-space: nowrap;
  transition: background-color 120ms ease, color 120ms ease;
}
.ui-as-menu__item[data-rich] { min-height: 44px; padding: 6px 10px; }
.ui-as-menu__item > svg { flex: none; color: var(--muted); }
.ui-as-menu__logo { display: grid; flex: none; place-items: center; width: 16px; height: 16px; }
.ui-as-menu__logo > svg { width: 14px; height: 14px; }
/* A flag: a small rounded rectangle with a hairline, so white stripes don't melt into the menu */
.ui-as-flag, .ui-as-menu__logo > .ui-as-flag { flex: none; width: 16px; height: 11px; overflow: hidden; border-radius: 2.5px; box-shadow: 0 0 0 0.5px rgba(0, 0, 0, 0.14); }
/* In the library's Dropdown Menu, whose logo slot draws marks square: a flag keeps its 16 × 11 */
.ui-dmc__icon[data-logo] > svg.ui-as-flag { width: 16px; height: 11px; }
.ui-as-menu__item > svg:last-child:not(:first-child) { margin-left: auto; color: var(--ink); }
.ui-as-menu__label { display: grid; flex: 1; gap: 1px; min-width: 0; }
.ui-as-menu__label small { font-size: 12px; line-height: 16px; color: var(--faint); }
/* Long names in menus stay on one line – the full name on hover */
.ui-as-menu__text, .ui-as-menu__label small { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.ui-as-menu__item > svg { flex: none; }
.ui-as-menu__empty { display: block; padding: 6px 8px; font-size: 13px; color: var(--faint); }
.ui-as-menu__item:hover { background: var(--hover); color: var(--ink); }
.ui-as-menu__item[aria-checked="true"] { color: var(--ink); font-weight: 500; }
.ui-as-menu__item[aria-checked="true"] small { font-weight: 400; }
.ui-as-menu__item[data-danger] { color: var(--aui-red, #f04438); }
/* Shortcuts, dividers and menus that open to the side */
.ui-as-menu__kbd { flex: none; margin-left: auto; padding-left: 16px; font: inherit; font-size: 12px; letter-spacing: 0.04em; color: var(--faint); }
.ui-as-menu__item > .ui-as-menu__kbd + svg { margin-left: 8px; }
.ui-as-menu__sep { display: block; height: 1px; margin: 4px 6px; background: var(--line); }
.ui-as-menu__sub { position: relative; display: block; }
.ui-as-menu__item > svg:last-child:not(:first-child):is(.ui-as-menu__sub svg) { color: var(--faint); }
.ui-as-menu__subpanel {
  position: absolute;
  z-index: 1;
  top: -4px;
  left: calc(100% + 4px);
  visibility: hidden;
  opacity: 0;
  transition: opacity 120ms ease, visibility 0s 120ms;
}
/* A bridge over the gap, so moving the pointer across doesn't close it */
.ui-as-menu__subpanel::before { content: ""; position: absolute; top: 0; bottom: 0; left: -8px; width: 8px; }
.ui-as-menu__sub:is(:hover, :focus-within, [data-open]) > .ui-as-menu__subpanel { visibility: visible; opacity: 1; transition: opacity 120ms ease; }
.ui-as-menu__sub:is(:hover, :focus-within, [data-open]) > .ui-as-menu__item { background: var(--hover); color: var(--ink); }
.ui-as-menu__item[data-danger] > svg { color: var(--aui-red, #f04438); }

.ui-as-toast {
  max-width: calc(100% - 32px);
  position: absolute;
  z-index: 40;
  left: 50%;
  bottom: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
  height: 36px;
  padding: 0 14px 0 12px;
  border-radius: 999px;
  background: var(--ink);
  font-size: 13px;
  color: var(--on-ink);
  box-shadow: 0 10px 24px -8px rgba(0, 0, 0, 0.35);
  translate: -50% 0;
  animation: ui-as-toast 2200ms var(--ease) both;
}

/* Shared motion: content that changes (tabs, filters, new messages) rises in; --i staggers a list */
.ui-as-rise { animation: ui-as-rise 420ms var(--ease) both; animation-delay: calc(var(--i, 0) * 35ms); }
/* The rise makes each row a layer of its own – a row with an open menu lifts above the rows after it */
.ui-as-rise:has(.ui-as-popover) { position: relative; z-index: 10; }
@keyframes ui-as-rise { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes ui-as-pop { from { opacity: 0; transform: scale(0.6); } to { opacity: 1; transform: none; } }
@keyframes ui-as-toast {
  0% { opacity: 0; transform: translateY(12px) scale(0.96); }
  12%, 85% { opacity: 1; transform: none; }
  100% { opacity: 0; transform: translateY(6px); }
}
@media (prefers-reduced-motion: reduce) {
  /* One 1ms run – a looping pulse (a live dot, a listening mic) doesn't turn into a flicker */
  .ui-as *, .ui-as *::before, .ui-as *::after { animation-duration: 1ms !important; animation-iteration-count: 1 !important; transition-duration: 1ms !important; scroll-behavior: auto !important; }
}
/* Rows in scrolling lists and menus: the focus ring sits inside the row, so the list's edge doesn't cut it off */
.ui-as :is(.ui-as-side__nav, .ui-as-menu, .ui-as-panel__body) :is(button, [href], [tabindex]):focus-visible { outline-offset: -2px; }
/* Keyboard focus on a menu item shows like the pointer's hover */
.ui-as-menu__item:focus-visible { background: var(--hover); color: var(--ink); }

/* Touch screens: no hover – the row tools stay in view (just ⋯; pinning is in its menu), and targets grow to a finger's size */
@media (hover: none) {
  .ui-as-chat-row__tools, .ui-as-chat-folder-row .ui-as-chat-row__tools { opacity: 1; }
  .ui-as-chat-row__pin { display: none; }
  .ui-as-chat-row .ui-as-chat-row__link, .ui-as-chat-folder-row > .ui-as-chat-folder { padding-right: 34px; }
  .ui-as-chat-row:is(:hover, :has(:focus-visible), [data-menu]) .ui-as-chat-row__link,
  .ui-as-chat-folder-row:is(:hover, :has(:focus-visible), [data-menu]) > .ui-as-chat-folder { padding-right: 34px; }
  .ui-as-chat-row__tool { color: var(--faint); }
}
@media (pointer: coarse) {
  /* Rows grow to a finger's height; small icon buttons keep their look and get an invisible ring that takes the tap */
  .ui-as-side__item, .ui-as-side__search, .ui-as-side__workspace, .ui-as-side__back, .ui-as-chat-row__rename { height: 36px; }
  .ui-as-chat-row__tools { top: 6px; }
  .ui-as-user { height: 40px; }
  .ui-as-menu__item { min-height: 40px; }
  :is(.ui-as-chat-row__tool, .ui-as-news__close, .ui-as-tool, .ui-asw__more, .ui-as-bar__pick) { position: relative; }
  .ui-as-icon-btn:not(.ui-as-side-open, .ui-as-panel-open) { position: relative; }
  :is(.ui-as-icon-btn, .ui-as-chat-row__tool, .ui-as-tool, .ui-as-view__btn, .ui-as-tabs__tab, .ui-asw__more)::after { content: ""; position: absolute; inset: -4px; }
  .ui-as-news__close::after { content: ""; position: absolute; inset: -7px; }
}

/* Empty state in a dashed drop area */

/* The library's Empty State, Skeleton and Spinner in the shell's own colors */
.ui-as .ui-as-empty {
  --ui-estc-title: var(--strong);
  --ui-estc-text: var(--muted);
  --ui-estc-icon: var(--muted);
  --ui-estc-tile: var(--chip);
  --ui-estc-frame: var(--frame);
  --ui-estc-line: var(--line);
}
/* div. – above the item (which has no margin), below a screen's own rule (e.g. margin-top: 0 in a grid) */
div.ui-as-empty { margin-top: 16px; }
div.ui-as-empty:where([data-bare]) { margin-top: 0; }
.ui-as .ui-as-skel { --ui-sklc-fill: var(--chip-hover); }
.ui-as-art-skel { display: flex; flex-direction: column; gap: 20px; padding: 14px 16px; }
.ui-as-art-skel__head { display: flex; align-items: center; gap: 10px; }
.ui-as-art-skel__name { display: flex; flex-direction: column; gap: 6px; }
.ui-as-art-skel__body { display: flex; flex-direction: column; gap: 12px; }
.ui-as-art-skel__gap { height: 4px; }

/* Glass Folder from the library at icon size; hovering the whole card or row opens it like hovering the folder itself */
.ui-as-folder-card__icon, .ui-as-folder-head__icon {
  position: relative;
  flex: none;
  width: 30px;
  height: 25px;
}
.ui-as-folder-card__icon .ui-fd, .ui-as-folder-head__icon .ui-fd {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 75px;
  translate: -50% -50%;
}
:is(.ui-as-folder-card, .ui-as-folder-head):hover .ui-fd-icon { transform: translate(-50%, -48%) scale(1.03); }
:is(.ui-as-folder-card, .ui-as-folder-head):hover .ui-fd-doc--left { transform: translate(32px, 70px) rotate(9deg); }
:is(.ui-as-folder-card, .ui-as-folder-head):hover .ui-fd-doc--right { transform: translate(286px, 76px) rotate(5deg); }
:is(.ui-as-folder-card, .ui-as-folder-head):hover .ui-fd-doc--middle { transform: translate(166px, 52px) rotate(-7deg); }
:is(.ui-as-folder-card, .ui-as-folder-head):hover .ui-fd-front {
  top: 24.8%;
  border-radius: 3cqw 3cqw 6.333cqw 6.333cqw;
  transform: translateZ(2.33cqw) rotateX(-9deg);
}

.ui-as-meta { display: inline-flex; align-items: center; gap: 4px; font-size: 12px; line-height: 16px; color: var(--muted); }
.ui-as-meta svg { color: var(--faint); }
.ui-as-toast > svg { flex: none; }
.ui-as-toast__text { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }


/* File card – used for templates and for the files inside a folder */
.ui-as-template { display: grid; grid-template-columns: minmax(0, 1fr); width: 239px; text-align: left; }
.ui-as-template__preview {
  display: block;
  height: 152px;
  padding: 3px;
  border-radius: calc(var(--aui-radius-lg, 10px) + 2px);
  background: var(--frame);
  box-shadow: inset 0 0 0 0.5px var(--line);
  transition: box-shadow 160ms ease;
}
.ui-as-template__preview .ui-as-doc { box-shadow: var(--card-shadow); }
.ui-as-template:hover .ui-as-template__preview { box-shadow: inset 0 0 0 0.5px var(--line-strong); }
/* One line at every width – the full name on hover */
.ui-as-template__name { margin-top: 8px; overflow: hidden; font-size: 13px; line-height: 20px; font-weight: 500; color: var(--strong); white-space: nowrap; text-overflow: ellipsis; }
.ui-as-template .ui-as-meta { margin-top: 0; }

/* File illustration: a small white page with hairline rows, a colored type tag over its lower left corner */
.ui-as-doc {
  --doc-tag: var(--aui-blue, #4fb0fe);
  --doc-tag-shade: 30, 64, 175;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  border-radius: var(--aui-radius-lg, 10px);
  background: var(--card);
}
.ui-as-doc[data-color="rose"] { --doc-tag: var(--aui-red, #f04438); --doc-tag-shade: 159, 18, 57; }
.ui-as-doc[data-color="emerald"] { --doc-tag: var(--aui-teal, #13b3a0); --doc-tag-shade: 6, 95, 70; }
/* Moved right by what the tag adds on the left, so the drawing sits in the middle */
.ui-as-doc__art { position: relative; display: block; width: 52px; translate: 4px 0; }

/* The page: white with a hairline ring and a soft shadow; the top right corner is cut (rounded where cuts aren't supported) */
.ui-as-doc__page {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  gap: 10px;
  /* The lower room the closing stroke had stays, so the page keeps its height */
  padding: 10px 10px 22px;
  border-radius: 5px 9px 5px 5px;
  corner-top-right-shape: bevel;
  background: var(--card);
  box-shadow: 0 0 0 1px var(--line-strong), 0 4px 8px -2px rgba(0, 0, 0, 0.065), 0 2px 3px -2px rgba(0, 0, 0, 0.065);
  transition: transform 420ms var(--ease);
}
.ui-as-doc__rows { display: flex; flex-direction: column; gap: 5px; }
.ui-as-doc__row { display: flex; gap: 3px; }
.ui-as-doc__row i { display: block; height: 2px; border-radius: 1px; background: color-mix(in srgb, var(--ink) 10%, transparent); }
/* A spreadsheet's first row is its header */
.ui-as-doc__page[data-type="XLS"] .ui-as-doc__row:first-child i { background: color-mix(in srgb, var(--ink) 22%, transparent); }
.ui-as-doc__image { display: block; height: 14px; border-radius: 2px; background: color-mix(in srgb, var(--ink) 6%, transparent); }
/* The type tag overlaps the page's lower left corner; the box is trimmed to the capitals, so they sit in its
   exact middle (a slightly deeper top padding stands in where trimming isn't supported) */
.ui-as-doc__type {
  position: absolute;
  left: -8px;
  bottom: 11px;
  z-index: 2;
  display: block;
  padding: 3.5px 4px 2.5px;
  border-radius: 3px;
  background: var(--doc-tag);
  box-shadow: 0 4px 8px -3px rgba(var(--doc-tag-shade), 0.3);
  font-size: 9px;
  line-height: 1;
  font-weight: 600;
  letter-spacing: 0.03em;
  color: #fff;
  transition: transform 420ms var(--ease);
}
@supports (text-box: trim-both cap alphabetic) {
  .ui-as-doc__type { padding: 3.5px 4px; text-box: trim-both cap alphabetic; }
}
.ui-as-template:hover .ui-as-doc__page { transform: translateY(-2px); }
.ui-as-template:hover .ui-as-doc__type { transform: translate(-1px, -3px); }

/* Workspace initial and user initials */
.ui-as-side__logo { font-size: 10.5px; font-weight: 500; color: var(--ink); }
.ui-as-user__initials {
  display: grid;
  place-items: center;
  width: 21px;
  height: 21px;
  border-radius: 50%;
  background: var(--accent-soft);
  font-size: 9.5px;
  font-weight: 500;
  color: var(--accent);
}

/* Scrim behind the drawer – only used on narrow screens */
.ui-as-scrim { display: none; }

/* ---------- Resizing ---------- */

/* A 12px strip centred on the main area's edge; on hover a darker line runs along the page's edge, fading to nothing
   at both ends – so it meets the rounded corners of the inset panel softly */
.ui-as-resizer {
  position: absolute;
  z-index: 15;
  top: calc(var(--bar-h, 0px) + var(--frame-gap));
  bottom: var(--frame-gap);
  width: 12px;
  cursor: col-resize;
  touch-action: none;
  outline: none;
}
.ui-as-resizer[data-edge="left"] { left: calc(var(--side-w, 272px) - 6px); }
/* Centred on the work area's left edge – inset it sits half a gap off the page, flat it touches it */
.ui-as-resizer[data-edge="right"] { right: calc(var(--panel-w, 360px) - var(--frame-gap) / 2 - 6px); }
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-resizer[data-edge="right"] { right: calc(var(--panel-w, 360px) - 6px); }
.ui-as-resizer::after {
  content: "";
  position: absolute;
  top: 0;
  bottom: 0;
  width: 1px;
  background: linear-gradient(to bottom, transparent, var(--edge-active) 22%, var(--edge-active) 78%, transparent);
  opacity: 0;
  transition: opacity 180ms ease;
}
/* Exactly on a border: the page's left edge on the left, the work area's left edge on the right */
.ui-as-resizer[data-edge="left"]::after { left: 6px; }
.ui-as-resizer[data-edge="right"]::after { left: 6px; }
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-resizer[data-edge="right"]::after { left: 5px; }
.ui-as-resizer:hover::after { opacity: 1; transition-delay: 100ms; }
.ui-as-resizer:focus-visible::after, .ui-as-resizer[data-dragging]::after { opacity: 1; transition-delay: 0ms; }

/* ---------- Work area on the right ---------- */

/* Sits on the gray page like the sidebar, mirrored; the main area gives up its right gap to it */
.ui-as-app[data-panel] > .ui-as-main { grid-column: 2; }
.ui-as-app[data-panel="open"] > .ui-as-main { margin-right: 0; }
/* Inset: the work area is a rounded white panel of its own, like the main area */
.ui-as-panel {
  grid-column: 3;
  grid-row: 1;
  display: flex;
  flex-direction: column;
  /* Half the gap towards the main area, the full one to the window's edges */
  width: calc(var(--panel-w, 360px) - 1.5 * var(--frame-gap));
  min-height: 0;
  margin: var(--frame-gap) var(--frame-gap) var(--frame-gap) calc(var(--frame-gap) / 2);
  padding: 10px 10px 10px 12px;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--radius-panel);
  background: var(--panel);
  transition: opacity 260ms ease, translate 420ms var(--ease), visibility 0s;
}
.ui-as-app[data-panel="closed"] > .ui-as-panel {
  visibility: hidden;
  opacity: 0;
  translate: 16px 0;
  transition: opacity 200ms ease, translate 420ms var(--ease), visibility 0s 420ms;
}
.ui-as-panel__head {
  display: flex;
  flex: none;
  align-items: center;
  justify-content: space-between;
  height: 28px;
  padding-left: 6px;
}
.ui-as-panel__title { display: flex; align-items: center; gap: 8px; min-width: 0; overflow: hidden; font-size: 13px; font-weight: 500; color: var(--ink); white-space: nowrap; text-overflow: ellipsis; }
/* A plain text heading needs a block box for its … */
.ui-as-panel__title:not(:has(> *)) { display: block; }
.ui-as-panel__body { flex: 1; min-height: 0; overflow-y: auto; overscroll-behavior: contain; margin: 0 -2px; padding: 12px 2px 4px; }

/* Its icon waits in the top right corner of the main area while it's closed */
.ui-as-panel-open {
  position: fixed;
  z-index: 20;
  top: calc(var(--frame-gap) + var(--as-head-edge) + (var(--as-head-h) - 28px) / 2);
  right: calc(var(--frame-gap) + 9px);
  visibility: hidden;
  opacity: 0;
  transition: opacity 200ms ease, visibility 0s 200ms;
}
.ui-as-app[data-panel="closed"] > .ui-as-panel-open {
  visibility: visible;
  opacity: 1;
  transition: opacity 260ms ease 160ms, visibility 0s;
}

/* ---------- Header across the app ---------- */

/* One bar over everything: a row of its own on top, split into the sidebar's, the page's and the work area's part */
.ui-as-app[data-header] { --bar-h: var(--as-head-h); grid-template-rows: var(--bar-h) minmax(0, 1fr); }
.ui-as-app[data-header] > .ui-as-bar--side { grid-area: 1 / 1; }
.ui-as-app[data-header] > .ui-as-bar--main { grid-area: 1 / 2; }
.ui-as-app[data-header] > .ui-as-bar--panel { grid-area: 1 / 3; }
.ui-as-app[data-header] > .ui-as-side { grid-area: 2 / 1; }
.ui-as-app[data-header] > .ui-as-main { grid-area: 2 / 2; margin-top: 0; }
.ui-as-app[data-header] > .ui-as-panel { grid-area: 2 / 3; }
.ui-as-bar {
  display: flex;
  align-items: center;
  gap: 2px;
  min-width: 0;
  height: var(--bar-h);
  overflow: hidden;
  transition: padding 420ms var(--ease), opacity 260ms ease;
}
.ui-as-bar--side { padding: 0 10px; }
.ui-as-bar--side > :last-child { margin-left: auto; }
.ui-as-bar--side > .ui-as-anchor { display: flex; min-width: 0; }
.ui-as-app[data-header] > .ui-as-side > .ui-as-side__search { margin-top: 0; }
.ui-as-bar--main { justify-content: space-between; gap: 12px; padding: 0 12px 0 18px; }
.ui-as-bar--panel { justify-content: space-between; padding: 0 10px 0 16px; }
.ui-as-bar__title { display: flex; align-items: center; gap: 6px; min-width: 0; font-size: 13px; font-weight: 500; color: var(--ink); white-space: nowrap; }
.ui-as-bar__title > * { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.ui-as-bar__actions { display: flex; flex: none; align-items: center; gap: 2px; }
.ui-as-bar__pick { display: flex; align-items: center; gap: 6px; min-width: 0; height: 30px; margin-left: -8px; padding: 0 8px; border-radius: calc(var(--aui-radius-sm, 5px) + 3px); color: var(--ink); }
.ui-as-bar__pick > span { overflow: hidden; text-overflow: ellipsis; }
.ui-as-bar__pick svg { color: var(--faint); }
.ui-as-bar__pick:hover { background: var(--hover); }
/* The sidebar's head – workspace and collapse – moves into the bar; the search field leads the sidebar */
.ui-as-app[data-header] > .ui-as-side > .ui-as-side__head,
.ui-as-app[data-header] > .ui-as-panel > .ui-as-panel__head { display: none; }
.ui-as-app[data-header] > .ui-as-panel > .ui-as-panel__body { padding-top: 2px; }
/* Folded away: the work area's part of the bar goes with it */
.ui-as-app[data-header][data-panel="closed"] > .ui-as-bar--panel { opacity: 0; }
.ui-as-app[data-header][data-panel="closed"] > .ui-as-bar--main { padding-right: 50px; }
.ui-as-app[data-header] > :is(.ui-as-side-open, .ui-as-panel-open) { top: calc((var(--bar-h) - 28px) / 2); }

/* A page that is just one centred block, e.g. an empty state */
.ui-as-center { display: grid; place-items: center; min-height: 100%; padding: 48px 32px; }

/* ---------- Double sidebar (rail), like Codex and ChatGPT ---------- */
/* A column of icons on the gray page, and the chats in a light gray column inside the main area, split by a line */
.ui-as-app[data-rail] { --rail-w: 52px; --side-col: calc(var(--rail-w) + var(--side-w, 272px)); }
.ui-as-app[data-rail] > .ui-as-side { flex-direction: row; width: 100%; padding: 0; overflow: hidden; }
.ui-as-app[data-rail] > .ui-as-resizer[data-edge="left"] { left: calc(var(--rail-w) + var(--side-w, 272px) - 6px); }
.ui-as-rail { display: flex; flex: none; flex-direction: column; align-items: center; gap: 2px; width: var(--rail-w); min-height: 0; padding: calc(var(--frame-gap) + 4px) 0 calc(var(--frame-gap) + 2px); overflow-y: auto; scrollbar-width: none; }
/* Many icons in a short window: the column scrolls, the account stays reachable at its end */
.ui-as-rail::-webkit-scrollbar { display: none; }
.ui-as-rail > .ui-as-anchor { display: flex; }
.ui-as-rail__logo { display: grid; place-items: center; width: 30px; height: 30px; margin-bottom: 4px; border-radius: 9px; background: var(--panel); box-shadow: 0 0 0 0.5px var(--line-strong), 0 1px 2px rgba(0, 0, 0, 0.05); font-size: 12px; font-weight: 600; color: var(--ink); transition: box-shadow 140ms ease; }
.ui-as-rail__logo:hover { box-shadow: 0 0 0 0.5px var(--line-strong), 0 2px 6px -2px rgba(0, 0, 0, 0.14); }
.ui-as-rail__item { position: relative; display: grid; flex: none; place-items: center; width: 34px; height: 34px; border-radius: calc(var(--aui-radius-sm, 5px) + 4px); color: var(--muted); transition: background-color 120ms ease, color 120ms ease; }
.ui-as-rail__item:hover { background: var(--hover); color: var(--ink); }
.ui-as-rail__item[aria-current="page"] { background: var(--select); color: var(--ink); }
.ui-as-rail__dot { position: absolute; top: 7px; right: 7px; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 0 2px var(--page); }
.ui-as-rail__line { flex: none; width: 18px; height: 1px; margin: 6px 0; background: var(--line); }
.ui-as-rail__spacer { flex: 1; }
.ui-as-rail .ui-as-rail__account > .ui-as-user { flex: none; justify-content: center; width: 34px; height: 34px; padding: 0; }
.ui-as-rail .ui-as-rail__account > .ui-as-user > svg, .ui-as-rail .ui-as-rail__account .ui-as-user__name { display: none; }
.ui-as-rail .ui-as-rail__account > .ui-as-user > :is(img, .ui-as-user__initials) { width: 24px; height: 24px; }
.ui-as-user-menu.ui-as-rail__account .ui-as-menu { width: 244px; }
/* The chats: the main area's left part – between the gray page and the white page in tone, a line on their right */
.ui-as-side__col {
  display: flex;
  flex: none;
  flex-direction: column;
  width: var(--side-w, 272px);
  min-height: 0;
  margin: var(--frame-gap) 0;
  padding: 10px 8px 8px 10px;
  border: 1px solid var(--line);
  border-radius: var(--radius-panel) 0 0 var(--radius-panel);
  background: var(--side-tint);
}
.ui-as-app[data-rail] > .ui-as-main { border-left: 0; border-top-left-radius: 0; border-bottom-left-radius: 0; }
.ui-as-side__col > .ui-as-side__nav { margin-top: 12px; }
.ui-as-project-menu .ui-as-menu { min-width: 220px; }

/* ---------- Narrow: tablet and phone ---------- */
/* Container queries on the app itself, so a scaled 1440 px preview keeps the desktop layout */

@container ui-as (max-width: 900px) {
  /* The main panel takes the whole width; the sidebar slides in over it as a drawer */
  .ui-as-app > .ui-as-main { grid-row: 1; grid-column: 1 / -1; margin-left: var(--frame-gap); }
  .ui-as-app > .ui-as-side {
    position: absolute;
    z-index: 50;
    top: 0;
    bottom: 0;
    left: 0;
    width: min(264px, calc(100% - 56px));
    padding: 10px 10px 10px 12px;
    border-radius: 0 calc(var(--aui-radius-lg, 10px) + 8px) calc(var(--aui-radius-lg, 10px) + 8px) 0;
    background: var(--page);
    box-shadow: 0 0 0 0.5px var(--line-strong), 0 24px 48px -12px rgba(0, 0, 0, 0.28);
  }
  .ui-as-app:not([data-drawer]) > .ui-as-side {
    visibility: hidden;
    translate: calc(-100% - 24px) 0;
    transition: translate 380ms var(--ease), visibility 0s 380ms;
  }
  .ui-as-app[data-drawer] > .ui-as-side {
    visibility: visible;
    opacity: 1;
    translate: 0 0;
    transition: translate 420ms var(--ease), visibility 0s;
  }
  /* The work area on the right becomes a drawer too, sliding in from the right */
  .ui-as-app[data-panel] { --panel-col: 0px; }
  .ui-as-app[data-panel] > .ui-as-main { grid-column: 1 / -1; margin-right: var(--frame-gap); }
  .ui-as-app > .ui-as-panel {
    /* Out of the grid: absolutely placed against the whole app, not the zero-width column it sits in on wide screens */
    grid-area: auto;
    position: absolute;
    z-index: 50;
    top: 0;
    right: 0;
    bottom: 0;
    width: min(360px, calc(100% - 56px));
    margin: 0;
    padding: 10px 12px 10px 12px;
    border: 0;
    border-radius: calc(var(--aui-radius-lg, 10px) + 8px) 0 0 calc(var(--aui-radius-lg, 10px) + 8px);
    background: var(--panel);
    box-shadow: 0 0 0 0.5px var(--line-strong), 0 24px 48px -12px rgba(0, 0, 0, 0.28);
  }
  .ui-as-app[data-panel="closed"] > .ui-as-panel {
    translate: calc(100% + 24px) 0;
    transition: translate 380ms var(--ease), visibility 0s 380ms;
  }
  .ui-as-app[data-panel="open"] > .ui-as-panel { visibility: visible; opacity: 1; translate: 0 0; transition: translate 420ms var(--ease), visibility 0s; }
  .ui-as-app[data-panel="open"] > .ui-as-scrim { visibility: visible; opacity: 1; transition: opacity 300ms ease, visibility 0s; }
  .ui-as-app > .ui-as-panel-open { right: calc(var(--frame-gap) + 9px); background: var(--panel); }
  .ui-as-app > .ui-as-panel-open:hover { background: var(--chip); }
  /* With the header: the page's part spans the bar; the sidebar's and work area's parts go into their drawers */
  .ui-as-app[data-header] > .ui-as-main { grid-area: 2 / 1 / 3 / -1; }
  .ui-as-app[data-header] > .ui-as-bar--main { grid-area: 1 / 1 / 2 / -1; padding-left: 50px; }
  .ui-as-app[data-header][data-panel] > .ui-as-bar--main { padding-right: 50px; }
  .ui-as-app[data-header] > :is(.ui-as-bar--side, .ui-as-bar--panel) { display: none; }
  /* The drawers cover the full height, over the bar too – not just their grid row */
  .ui-as-app[data-header] > :is(.ui-as-side, .ui-as-panel) { grid-area: auto; }
  .ui-as-app[data-header] > .ui-as-side > .ui-as-side__head { display: flex; }
  .ui-as-app[data-header] > .ui-as-side > .ui-as-side__search { margin-top: 12px; }
  .ui-as-app[data-header] > .ui-as-panel > .ui-as-panel__head { display: flex; }
  .ui-as-app[data-header] > .ui-as-panel > .ui-as-panel__body { padding-top: 12px; }
  .ui-as-app > .ui-as-scrim {
    display: block;
    position: absolute;
    z-index: 45;
    inset: 0;
    background: rgba(0, 0, 0, 0.28);
    visibility: hidden;
    opacity: 0;
    transition: opacity 300ms ease, visibility 0s 300ms;
  }
  .ui-as-app[data-drawer] > .ui-as-scrim { visibility: visible; opacity: 1; transition: opacity 300ms ease, visibility 0s; }
  :where(.dark, [data-theme="dark"]) .ui-as-app > .ui-as-scrim { background: rgba(0, 0, 0, 0.55); }
  /* The icon to open it always waits in the top left corner of the panel */
  .ui-as-app:not([data-drawer]) > .ui-as-side-open {
    visibility: visible;
    opacity: 1;
    transition: opacity 260ms ease 160ms, visibility 0s;
  }
  /* It stays put while the panel scrolls, so it gets the panel's color behind it */
  .ui-as-app > .ui-as-side-open { left: calc(var(--frame-gap) + 9px); background: var(--panel); }
  .ui-as-app > .ui-as-side-open:hover { background: var(--chip); }

  /* Page parts: less padding, toolbars wrap */
  .ui-as-page { padding: 46px 24px 48px; }
  /* File cards fill the cells of the grids the screens switch to */
  .ui-as-main .ui-as-file-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); }
  .ui-as-main .ui-as-file-grid .ui-as-template { width: 100%; }
  .ui-as-main .ui-as-file-grid .ui-as-template__name { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .ui-as-main .ui-as-file-grid .ui-as-meta { display: block; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .ui-as-main .ui-as-file-grid .ui-as-meta svg { display: inline-block; vertical-align: -2px; }
  .ui-as-toolbar { flex-wrap: wrap; row-gap: 10px; }
  .ui-as-toolbar__right { flex-wrap: wrap; }
  .ui-as-section-head[data-inline] { flex-wrap: wrap; row-gap: 12px; }
}

@container ui-as (max-width: 600px) {
  .ui-as-app > * { --frame-gap: 6px; }
  /* No room for a submenu beside its menu on a phone: it opens in place, under its item */
  .ui-as-menu__subpanel { position: static; display: none; min-width: 0 !important; margin: 2px 0 4px; padding: 0 0 0 14px; border-radius: 0; background: none; box-shadow: none; }
  .ui-as-menu__subpanel::before { display: none; }
  .ui-as-menu__sub:is(:focus-within, [data-open]) > .ui-as-menu__subpanel { display: block; }
  .ui-as-menu__sub:is(:focus-within, [data-open]) > .ui-as-menu__item > svg:last-child { transform: rotate(90deg); }
  /* A short screen keeps the sidebar for its lists: What's new waits for a wider one */
  .ui-as-news { display: none; }
  .ui-as-app > .ui-as-main { border-radius: calc(var(--aui-radius-lg, 10px) + 1px); }
  .ui-as-page { padding: 52px 16px 40px; }
  .ui-as-page-title { font-size: 20px; line-height: 26px; }
  .ui-as-section-head { margin-top: 32px; }
  /* Toolbars: tabs on their own line, the tools below; search takes what is left */
  .ui-as-toolbar__left, .ui-as-toolbar__right { width: 100%; }
  .ui-as-toolbar__right .ui-as-search { order: 9; flex: 1 1 100%; width: auto !important; }
  .ui-as-toolbar__right > .ui-buttonc:last-child { margin-left: auto; }
  .ui-as-tabs { max-width: 100%; overflow-x: auto; scrollbar-width: none; }
  .ui-as-tabs__tab { flex: none; }
  /* div. – above the item's own padding, below a screen's own rule (e.g. Folder's) */
  div.ui-as-empty { padding: 36px 18px 40px; }
  .ui-as-popover--filter .ui-fmc { width: 300px; }
  .ui-as-main .ui-as-file-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px 10px; }
  .ui-as-main .ui-as-file-grid .ui-as-template__preview { height: 128px; }
}

/* ----- Flat: no panel – the page runs edge to edge next to the sidebar, split by a hairline ----- */
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) { --frame-gap: 0px; --as-head-edge: 0px; }
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > * { --frame-gap: 0px; --as-head-edge: 0px; }

/* Main area: no panel – it fills the rest of the app, a hairline on its left is the only edge */
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-main {
  margin: 0;
  border: 0;
  border-left: 1px solid var(--line);
  border-radius: 0;
  background: var(--panel);
}
/* Sidebar collapsed: nothing is left to divide */


/* Work area on the right: flat too – a hairline on the main area's right is its edge */
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"]))[data-panel="open"] > .ui-as-main { border-right: 1px solid var(--line); }
/* …and the same white as the main area – no frame of its own */
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-panel { width: var(--panel-w, 360px); margin: 0; padding: 10px 10px 10px 10px; border: 0; border-radius: 0; background: var(--panel); }
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-panel-open { right: 12px; }

/* Header across the app: the page's and the work area's parts are white like the page, the sidebar's stays gray; no line under it –
   only the page's side lines run on up through the bar */
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"]))[data-header] > .ui-as-bar--main { background: var(--panel); border-left: 1px solid var(--line); transition: padding 420ms var(--ease), border-color 260ms ease; }

.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"]))[data-header][data-panel="open"] > .ui-as-bar--main { border-right: 1px solid var(--line); }
/* The work area's part of the bar is white like the page's – only the sidebar's part stays gray */
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"]))[data-header] > .ui-as-bar--panel { background: var(--panel); }
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"]))[data-header] > .ui-as-side { padding-top: 2px; }
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"]))[data-header] > .ui-as-panel { padding-top: 12px; }

/* The sidebar's rows line up with a page's 46px title on top */
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-side { padding: 10px 10px 10px 10px; }

/* The icon that brings the sidebar back sits where the collapse icon was */
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-side-open { left: 12px; }

@container ui-as (max-width: 900px) {
  /* Narrow: the main area is the whole app; the drawer slides in flat with a hairline edge */
  .ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-main { margin: 0; border-left: 0; }
  .ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-side {
    border-radius: 0;
    box-shadow: 1px 0 0 var(--line), 0 24px 48px -12px rgba(0, 0, 0, 0.22);
  }
  .ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-side-open { left: 12px; }
  .ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"]))[data-panel] > .ui-as-main { margin: 0; border-right: 0; }
  .ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-panel {
    border-radius: 0;
    box-shadow: -1px 0 0 var(--line), 0 24px 48px -12px rgba(0, 0, 0, 0.22);
  }
  .ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-panel-open { right: 12px; }
  .ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"]))[data-header] > .ui-as-bar--main { border-left: 0; border-right: 0; }
}
@container ui-as (max-width: 600px) {
  .ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-main { border-radius: 0; }
}

/* ----- Work area with tabs, like Codex: a white work area extended upward in light gray – the tabs sit on the gray,
   the active one a white pill, and the white sheet below runs to the panel's edges with only its top corners rounded ----- */
.ui-as-app > .ui-as-panel[data-ws] {
  --asw-tray: linear-gradient(rgba(0, 0, 0, 0.012), rgba(0, 0, 0, 0.012)), var(--panel);
  padding: 0;
  background: var(--asw-tray);
}
:where(.dark, [data-theme="dark"]) .ui-as-app > .ui-as-panel[data-ws] { --asw-tray: linear-gradient(rgba(0, 0, 0, 0.12), rgba(0, 0, 0, 0.12)), var(--panel); }
.ui-as-app:is([data-layout="flat"], [data-shell="flat"] :not([data-layout="inset"])) > .ui-as-panel[data-ws] { padding: 0; background: var(--asw-tray); }
.ui-as-app > .ui-as-panel[data-ws] > .ui-as-panel__head { display: flex; height: var(--as-head-h); padding: 0 6px 0 4px; }
/* The white sheet runs to the panel's edges – only its top corners round off under the tabs; it scrolls up and down
   only – wide content like a sheet scrolls sideways inside itself */
.ui-as-app > .ui-as-panel[data-ws] > .ui-as-panel__body { overflow-x: hidden; overscroll-behavior: none; margin: 0; padding: 16px 14px 14px; border-radius: var(--aui-radius-lg, 10px) var(--aui-radius-lg, 10px) 0 0; background: var(--card); box-shadow: 0 -0.5px 0 var(--line), 0 -1px 2px rgba(0, 0, 0, 0.03); }
/* With the header across the app the tabs stand in the bar's row, at the work area's top */
@container ui-as (min-width: 901px) {
  .ui-as-app[data-header] > .ui-as-panel[data-ws] { grid-area: 1 / 3 / 3 / 4; }
}
.ui-asw-bar { display: flex; flex: 0 1 auto; align-items: center; gap: 2px; min-width: 0; }
.ui-asw-ptabs { display: flex; align-items: center; gap: 2px; min-width: 0; padding: 2px; overflow-x: auto; overscroll-behavior: none; scrollbar-width: none; font-weight: 400; }
/* More tabs behind the end: it fades out */
.ui-asw-ptabs[data-more] { mask-image: linear-gradient(90deg, #000 calc(100% - 16px), transparent); }
.ui-asw-ptabs::-webkit-scrollbar { display: none; }
.ui-asw-ptab-wrap { position: relative; display: flex; flex: 0 1 auto; min-width: 92px; border-radius: calc(var(--aui-radius-sm, 5px) + 2px); transition: background-color 140ms ease; }
.ui-asw-ptab { display: inline-flex; flex: 0 1 auto; align-items: center; gap: 6px; width: 100%; min-width: 0; max-width: 156px; height: 28px; padding: 0 26px 0 10px; border-radius: calc(var(--aui-radius-sm, 5px) + 2px); font-size: 13px; color: var(--muted); transition: background-color 140ms ease, color 140ms ease; }
.ui-asw-ptab > svg { flex: none; color: var(--muted); }
.ui-asw-ptab-wrap:hover > .ui-asw-ptab { color: var(--ink); }
.ui-asw-ptab-wrap:not([data-on]):hover { background: var(--hover); }
/* The active tab: a white pill on the gray */
.ui-asw-ptab-wrap[data-on] { background: var(--card); box-shadow: var(--card-shadow); }
.ui-asw-ptab-wrap[data-on] > .ui-asw-ptab { color: var(--ink); }
/* A faint vertical line between tabs, like Codex – it steps aside next to the active or hovered tab */
.ui-asw-ptab-wrap + .ui-asw-ptab-wrap::before { content: ""; position: absolute; top: 8px; bottom: 8px; left: -1.5px; width: 1px; background: var(--line-strong); pointer-events: none; transition: opacity 140ms ease; }
.ui-asw-ptab-wrap:is([data-on], :hover)::before, .ui-asw-ptab-wrap:is([data-on], :hover) + .ui-asw-ptab-wrap::before { opacity: 0; }
.ui-asw-ptab:focus-visible { outline-offset: -2px; }
/* A name that doesn't fit fades out at its end instead of an … – the fade sits in its right padding, so a name that
   fits shows whole; the × has 28px of its own on the right, the name and its fade end before it */
.ui-asw-ptab__name { min-width: 0; padding-right: 8px; overflow: hidden; white-space: nowrap; mask-image: linear-gradient(90deg, #000 calc(100% - 12px), transparent); }
.ui-asw-ptab__name > * { display: inline; }
.ui-asw-ptab__count { flex: none; margin-left: -6px; padding: 0 5px; border-radius: calc(var(--aui-radius-sm, 5px) - 1px); background: var(--chip); font-size: 11px; font-weight: 500; line-height: 16px; color: var(--muted); font-variant-numeric: tabular-nums; }
.ui-asw-ptab__close { position: absolute; top: 6px; right: 5px; display: grid; place-items: center; width: 16px; height: 16px; border-radius: calc(var(--aui-radius-sm, 5px) - 1px); color: var(--muted); opacity: 0; transition: opacity 120ms ease, background-color 120ms ease; }
.ui-asw-ptab-wrap:is(:hover, [data-on], :focus-within) > .ui-asw-ptab__close { opacity: 1; }
.ui-asw-ptab__close:hover { background: var(--hover); color: var(--ink); }
/* + at the end: a New tab */
.ui-asw-ptab-new { display: grid; flex: none; place-items: center; width: 28px; height: 28px; border-radius: calc(var(--aui-radius-sm, 5px) + 2px); color: var(--muted); transition: background-color 140ms ease, color 140ms ease; }
.ui-asw-ptab-new:hover { background: var(--hover); color: var(--ink); }
.ui-asw-ptab-new:focus-visible { outline-offset: -2px; }
@media (pointer: coarse) {
  .ui-asw-ptab__close { opacity: 1; }
  .ui-asw-ptab-new { width: 36px; height: 36px; }
}
@media (prefers-reduced-motion: reduce) {
  .ui-asw-ptab-wrap, .ui-asw-ptab, .ui-asw-ptab__close, .ui-asw-ptab-new { transition: none; }
}

/* Narrow: the icons and the chats slide in together as one drawer; the page is a whole panel again */
@container ui-as (max-width: 900px) {
  .ui-as-app[data-rail] > .ui-as-side { width: min(320px, calc(100% - 40px)); padding: 0; }
  .ui-as-app[data-rail] .ui-as-side__col { flex: 1; width: auto; min-width: 0; margin: 0; padding: 10px 10px 10px 4px; border: 0; border-radius: 0; background: none; }
  .ui-as-app[data-rail] > .ui-as-main { border-left: 1px solid var(--line); border-radius: var(--radius-panel); }
}
`;
