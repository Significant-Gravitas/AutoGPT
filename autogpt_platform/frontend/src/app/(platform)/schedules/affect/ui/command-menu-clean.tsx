"use client";

import {
  memo,
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
  type SyntheticEvent,
} from "react";

/*
 * Command Menu (Clean) – search and run anything from one field, in a single self-contained file.
 * Clean style: no tray – one white panel with hairline shadows, small uppercase group labels, quiet key chips
 * and a last row with hints under a hairline.
 * Actions that use another tool carry its logo (Gmail, Slack, ClickUp …) and a ⌘1–⌘5 shortcut;
 * everyday commands use simple icons. Typing filters every group and offers to ask the assistant about it.
 * ↑ ↓ move row by row, Tab jumps section by section, ↵ runs, ⌘1–⌘5 run a suggestion straight away,
 * Esc clears the search or closes, ⌘K (Ctrl K on Windows) focuses the field from anywhere.
 * The styles are injected below, so no extra files are needed.
 */

export type Command = {
  id: string;
  label: string;
  /** Lighter text after the label, e.g. a person, a channel or a page */
  hint?: string;
  /** "logo" sits on a small white tile; "icon" is a plain glyph */
  icon?: ReactNode;
  logo?: ReactNode;
  /** Keys shown on the right, e.g. ["G", "I"] – suggestions get ⌘1–⌘5 automatically */
  keys?: string[];
  /** Extra words that should find this command */
  keywords?: string;
};

export type CommandGroup = { id: string; label: string; commands: Command[] };

export type CommandMenuProps = {
  groups?: CommandGroup[];
  placeholder?: string;
  /** Called with the command id; for the assistant fallback with "ask" and the typed text */
  onRun?: (id: string, query?: string) => void;
  /** Runs on Esc in an empty search */
  onClose?: () => void;
  /** Focus the field when the menu appears */
  autoFocus?: boolean;
  className?: string;
};

/* ---------- Icons ---------- */

const Glyph = ({ d }: { d: string }) => (
  <svg viewBox="0 0 24 24" className="ui-cmc__glyph" aria-hidden="true">
    <path d={d} />
  </svg>
);
const icons = {
  doc: "M7 3h7l5 5v13H7zM14 3v5h5M10 13h6M10 17h4",
  person:
    "M12 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8zM4.5 20a7.5 7.5 0 0 1 15 0M19 8v4M17 10h4",
  inbox: "M4 13l2.5-8h11L20 13v6H4zM4 13h4.5a3.5 3.5 0 0 0 7 0H20",
  home: "M4 11l8-7 8 7v9h-5v-6H9v6H4z",
  gear: "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6zM19.4 13.5l1.6 1.2-2 3.4-1.9-.7a7 7 0 0 1-1.7 1L15 20.5h-4l-.4-2.1a7 7 0 0 1-1.7-1l-1.9.7-2-3.4 1.6-1.2a7 7 0 0 1 0-3l-1.6-1.2 2-3.4 1.9.7a7 7 0 0 1 1.7-1L11 3.5h4l.4 2.1a7 7 0 0 1 1.7 1l1.9-.7 2 3.4-1.6 1.2a7 7 0 0 1 0 3z",
  moon: "M20 14.5A8 8 0 1 1 9.5 4a6.5 6.5 0 0 0 10.5 10.5z",
  help: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .9-1 1.6v.6M12 17h.01",
  upload: "M12 16V4M7 9l5-5 5 5M5 20h14",
  chart: "M5 20V10M12 20V4M19 20v-7",
};

const SparkTile = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <defs>
      <linearGradient id="ui-cmc-spark" x1="0" y1="0" x2="1" y2="1">
        <stop stopColor="#7C5CFF" />
        <stop offset="1" stopColor="#2F9BFF" />
      </linearGradient>
    </defs>
    <path
      fill="url(#ui-cmc-spark)"
      d="M12 2.5c.6 3.9 2.1 6.4 4.4 7.6 1.1.6 2.4.9 4.1 1.1v1.6c-4.6.6-7.3 3-8.5 8.7h-1.6c-.6-3.9-2.1-6.4-4.4-7.6-1.1-.6-2.4-.9-4.1-1.1v-1.6c4.6-.6 7.3-3 8.5-8.7z"
    />
  </svg>
);

/* Brand marks – Simple Icons (CC0) and the products' own colors */
/** Unique ids for gradients, so several logos on one page don't clash */
const useSvgId = () => "lg" + useId().replace(/[^a-zA-Z0-9_-]/g, "");

const GmailLogo = () => {
  const id = useSvgId();
  return (
    <svg viewBox="0 0 56.4108 44.8722" fill="none" aria-hidden="true">
      <path
        d="M44.2255 5.76727H56.4051V41.0239C56.4051 43.148 54.683 44.8701 52.5589 44.8701H46.1486C45.8961 44.8701 45.646 44.8204 45.4127 44.7237C45.1794 44.6271 44.9674 44.4854 44.7888 44.3069C44.6102 44.1283 44.4686 43.9163 44.3719 43.683C44.2753 43.4497 44.2255 43.1996 44.2255 42.947V5.76727Z"
        fill={`url(#${id}-0)`}
      />
      <path
        d="M12.1796 5.76727H0V41.0239C0 43.148 1.72213 44.8701 3.84618 44.8701H10.2565C10.509 44.8701 10.7591 44.8204 10.9924 44.7237C11.2257 44.6271 11.4377 44.4854 11.6163 44.3069C11.7949 44.1283 11.9365 43.9163 12.0332 43.683C12.1298 43.4497 12.1796 43.1996 12.1796 42.947V5.76727Z"
        fill="#FC413D"
      />
      <path
        d="M10.0085 1.42778C7.43377 -0.736342 3.5924 -0.403647 1.42828 2.17105C-0.735843 4.74543 -0.403148 8.58681 2.17155 10.7512L26.5557 31.2479C27.0178 31.6362 27.602 31.8492 28.2056 31.8492C28.8091 31.8492 29.3934 31.6362 29.8554 31.2479L54.2396 10.7509C56.814 8.58681 57.1467 4.74543 54.9825 2.17073C52.8184 -0.403648 48.977 -0.736342 46.4027 1.42778L28.2054 16.724L10.0085 1.42778Z"
        fill={`url(#${id}-1)`}
      />
      <defs>
        <linearGradient
          id={`${id}-0`}
          x1="50.3153"
          y1="5.76727"
          x2="50.3153"
          y2="44.8701"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#60D673" />
          <stop offset="0.17" stopColor="#42C868" />
          <stop offset="0.39" stopColor="#0EBC5F" />
          <stop offset="0.62" stopColor="#00A9BB" />
          <stop offset="0.86" stopColor="#3C90FF" />
          <stop offset="1" stopColor="#3186FF" />
        </linearGradient>
        <linearGradient
          id={`${id}-1`}
          x1="6.04194e-05"
          y1="6.45153"
          x2="56.4107"
          y2="6.45153"
          gradientUnits="userSpaceOnUse"
        >
          <stop offset="0.08" stopColor="#FF63A0" />
          <stop offset="0.3" stopColor="#FC413D" />
          <stop offset="0.5" stopColor="#FC413D" />
          <stop offset="0.65" stopColor="#FC413D" />
          <stop offset="0.72" stopColor="#FC5C30" />
          <stop offset="0.86" stopColor="#FEB10C" />
          <stop offset="0.91" stopColor="#FEC700" />
          <stop offset="0.96" stopColor="#FFDB0F" />
        </linearGradient>
      </defs>
    </svg>
  );
};

const SlackLogo = () => (
  <svg viewBox="0 0 49.883 49.877" fill="none" aria-hidden="true">
    <g transform="translate(0 0)">
      <path
        d="M18.3549 10.4784C15.4713 10.4784 13.1157 8.12279 13.1157 5.2392C13.1157 2.35561 15.4713 0 18.3549 0C21.2385 0 23.5941 2.35561 23.5941 5.2392V10.4784H18.3549Z"
        fill="#36C5F0"
      />
      <path
        d="M18.3575 13.12C21.2411 13.12 23.5967 15.4756 23.5967 18.3592C23.5967 21.2428 21.2411 23.5984 18.3575 23.5984H5.2392C2.35561 23.5984 0 21.2428 0 18.3592C0 15.4756 2.35561 13.12 5.2392 13.12H18.3575Z"
        fill="#36C5F0"
      />
    </g>
    <g transform="translate(26.28 0)">
      <path
        d="M13.1244 18.3592C13.1244 15.4756 15.48 13.12 18.3636 13.12C21.2472 13.12 23.6028 15.4756 23.6028 18.3592C23.6028 21.2428 21.2472 23.5984 18.3636 23.5984H13.1244V18.3592Z"
        fill="#2EB67D"
      />
      <path
        d="M10.4785 18.3575C10.4785 21.2411 8.12286 23.5967 5.23927 23.5967C2.35568 23.5967 7.4362e-05 21.2411 7.4362e-05 18.3575V5.2392C7.4362e-05 2.35561 2.35568 0 5.23927 0C8.12286 0 10.4785 2.35561 10.4785 5.2392V18.3575Z"
        fill="#2EB67D"
      />
    </g>
    <g transform="translate(0 26.28)">
      <path
        d="M10.4784 5.2392C10.4784 8.12279 8.12279 10.4784 5.2392 10.4784C2.35561 10.4784 0 8.12279 0 5.2392C0 2.35561 2.35561 0 5.2392 0H10.4784V5.2392Z"
        fill="#E01E5A"
      />
      <path
        d="M13.1157 5.2392C13.1157 2.35561 15.4713 0 18.3549 0C21.2385 0 23.5941 2.35561 23.5941 5.2392V18.3575C23.5941 21.2411 21.2385 23.5967 18.3549 23.5967C15.4713 23.5967 13.1157 21.2411 13.1157 18.3575V5.2392Z"
        fill="#E01E5A"
      />
    </g>
    <g transform="translate(26.28 26.28)">
      <path
        d="M5.23927 13.1157C8.12286 13.1157 10.4785 15.4713 10.4785 18.3549C10.4785 21.2385 8.12286 23.5941 5.23927 23.5941C2.35568 23.5941 7.4362e-05 21.2385 7.4362e-05 18.3549V13.1157H5.23927Z"
        fill="#ECB22E"
      />
      <path
        d="M5.23927 10.4784C2.35568 10.4784 7.4362e-05 8.12279 7.4362e-05 5.2392C7.4362e-05 2.35561 2.35568 0 5.23927 0H18.3576C21.2412 0 23.5968 2.35561 23.5968 5.2392C23.5968 8.12279 21.2412 10.4784 18.3576 10.4784H5.23927Z"
        fill="#ECB22E"
      />
    </g>
  </svg>
);

const ClickUpLogo = () => {
  const id = useSvgId();
  return (
    <svg viewBox="0 0 49.7096 59.6884" fill="none" aria-hidden="true">
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M0 45.9002L9.1618 38.8247C14.0602 45.1745 19.2307 48.1679 24.9455 48.1679C30.6603 48.1679 35.7401 45.2652 40.3663 38.9154L49.7096 45.8094C42.997 54.8805 34.6516 59.6882 24.9455 59.6882C15.3301 59.6882 6.89403 54.8805 0 45.9002Z"
        fill={`url(#${id}-0)`}
      />
      <path
        fillRule="evenodd"
        clipRule="evenodd"
        d="M24.9535 15.3301L8.62558 29.3903L1.09657 20.5914L25.0442 0L48.8105 20.5914L41.1908 29.2996L24.9535 15.3301Z"
        fill={`url(#${id}-1)`}
      />
      <defs>
        <linearGradient
          id={`${id}-0`}
          x1="0"
          y1="49.2666"
          x2="49.75"
          y2="49.2666"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#8930FD" />
          <stop offset="1" stopColor="#49CCF9" />
        </linearGradient>
        <linearGradient
          id={`${id}-1`}
          x1="1.09231"
          y1="14.6901"
          x2="48.7603"
          y2="14.6901"
          gradientUnits="userSpaceOnUse"
        >
          <stop stopColor="#FF02F0" />
          <stop offset="1" stopColor="#FFC800" />
        </linearGradient>
      </defs>
    </svg>
  );
};

const HubSpotLogo = () => (
  <svg viewBox="0 0 24 24" fill="#FF7A59" aria-hidden="true">
    <path d="M18.164 7.93V5.084a2.198 2.198 0 001.267-1.978v-.067A2.2 2.2 0 0017.238.845h-.067a2.2 2.2 0 00-2.193 2.193v.067a2.196 2.196 0 001.252 1.973l.013.006v2.852a6.22 6.22 0 00-2.969 1.31l.012-.01-7.828-6.095A2.497 2.497 0 104.3 4.656l-.012.006 7.697 5.991a6.176 6.176 0 00-1.038 3.446c0 1.343.425 2.588 1.147 3.607l-.013-.02-2.342 2.343a1.968 1.968 0 00-.58-.095h-.002a2.033 2.033 0 102.033 2.033 1.978 1.978 0 00-.1-.595l.005.014 2.317-2.317a6.247 6.247 0 104.782-11.134l-.036-.005zm-.964 9.378a3.206 3.206 0 113.215-3.207v.002a3.206 3.206 0 01-3.207 3.207z" />
  </svg>
);
const CalendarLogo = () => (
  <svg viewBox="0 0 24 24" fill="#4285F4" aria-hidden="true">
    <path d="M18.316 5.684H24v12.632h-5.684V5.684zM5.684 24h12.632v-5.684H5.684V24zM18.316 5.684V0H1.895A1.894 1.894 0 0 0 0 1.895v16.421h5.684V5.684h12.632zm-7.207 6.25v-.065c.272-.144.5-.349.687-.617s.279-.595.279-.982c0-.379-.099-.72-.3-1.025a2.05 2.05 0 0 0-.832-.714 2.703 2.703 0 0 0-1.197-.257c-.6 0-1.094.156-1.481.467-.386.311-.65.671-.793 1.078l1.085.452c.086-.249.224-.461.413-.633.189-.172.445-.257.767-.257.33 0 .602.088.816.264a.86.86 0 0 1 .322.703c0 .33-.12.589-.36.778-.24.19-.535.284-.886.284h-.567v1.085h.633c.407 0 .748.109 1.02.327.272.218.407.499.407.843 0 .336-.129.614-.387.832s-.565.327-.924.327c-.351 0-.651-.103-.897-.311-.248-.208-.422-.502-.521-.881l-1.096.452c.178.616.505 1.082.977 1.401.472.319.984.478 1.538.477a2.84 2.84 0 0 0 1.293-.291c.382-.193.684-.458.902-.794.218-.336.327-.72.327-1.149 0-.429-.115-.797-.344-1.105a2.067 2.067 0 0 0-.881-.689zm2.093-1.931l.602.913L15 10.045v5.744h1.187V8.446h-.827l-2.158 1.557zM22.105 0h-3.289v5.184H24V1.895A1.894 1.894 0 0 0 22.105 0zm-3.289 23.5l4.684-4.684h-4.684V23.5zM0 22.105C0 23.152.848 24 1.895 24h3.289v-5.184H0v3.289z" />
  </svg>
);
const NotionLogo = () => (
  <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
    <path d="M4.459 4.208c.746.606 1.026.56 2.428.466l13.215-.793c.28 0 .047-.28-.046-.326L17.86 1.968c-.42-.326-.981-.7-2.055-.607L3.01 2.295c-.466.046-.56.28-.374.466zm.793 3.08v13.904c0 .747.373 1.027 1.214.98l14.523-.84c.841-.046.935-.56.935-1.167V6.354c0-.606-.233-.933-.748-.887l-15.177.887c-.56.047-.747.327-.747.933zm14.337.745c.093.42 0 .84-.42.888l-.7.14v10.264c-.608.327-1.168.514-1.635.514-.748 0-.935-.234-1.495-.933l-4.577-7.186v6.952L12.21 19s0 .84-1.168.84l-3.222.186c-.093-.186 0-.653.327-.746l.84-.233V9.854L7.822 9.76c-.094-.42.14-1.026.793-1.073l3.456-.233 4.764 7.279v-6.44l-1.215-.139c-.093-.514.28-.887.747-.933zM1.936 1.035l13.31-.98c1.634-.14 2.055-.047 3.082.7l4.249 2.986c.7.513.934.653.934 1.213v16.378c0 1.026-.373 1.634-1.68 1.726l-15.458.934c-.98.047-1.448-.093-1.962-.747l-3.129-4.06c-.56-.747-.793-1.306-.793-1.96V2.667c0-.839.374-1.54 1.447-1.632z" />
  </svg>
);

/** Sample commands – replace with your own */
export const defaultGroups: CommandGroup[] = [
  {
    id: "suggestions",
    label: "Suggestions",
    commands: [
      {
        id: "email",
        label: "Write an email",
        hint: "Gmail",
        logo: <GmailLogo />,
        keywords: "mail send compose",
      },
      {
        id: "ask",
        label: "Ask assistant",
        hint: "Summaries, drafts, answers",
        logo: <SparkTile />,
        keywords: "ai chat question",
      },
      {
        id: "task",
        label: "Create a task",
        hint: "ClickUp",
        logo: <ClickUpLogo />,
        keywords: "todo issue",
      },
      {
        id: "crm",
        label: "Add contact to CRM",
        hint: "HubSpot",
        logo: <HubSpotLogo />,
        keywords: "lead deal company",
      },
      {
        id: "meet",
        label: "Schedule a meeting",
        hint: "Google Calendar",
        logo: <CalendarLogo />,
        keywords: "call invite event",
      },
    ],
  },
  {
    id: "share",
    label: "Share",
    commands: [
      {
        id: "slack",
        label: "Post an update",
        hint: "#team-updates",
        logo: <SlackLogo />,
        keywords: "message channel",
      },
      {
        id: "notion",
        label: "Save as page",
        hint: "Notion · Meeting notes",
        logo: <NotionLogo />,
        keywords: "doc wiki",
      },
    ],
  },
  {
    id: "general",
    label: "General",
    commands: [
      {
        id: "new-doc",
        label: "New document",
        icon: <Glyph d={icons.doc} />,
        keys: ["N"],
      },
      {
        id: "invite",
        label: "Invite teammates",
        icon: <Glyph d={icons.person} />,
        keys: ["I"],
      },
      {
        id: "upload",
        label: "Upload files",
        icon: <Glyph d={icons.upload} />,
        keys: ["U"],
      },
      {
        id: "theme",
        label: "Switch to dark mode",
        icon: <Glyph d={icons.moon} />,
        keywords: "theme appearance light",
        keys: ["⇧", "D"],
      },
    ],
  },
  {
    id: "go",
    label: "Go to",
    commands: [
      {
        id: "home",
        label: "Home",
        icon: <Glyph d={icons.home} />,
        keys: ["G", "H"],
      },
      {
        id: "inbox",
        label: "Inbox",
        icon: <Glyph d={icons.inbox} />,
        keys: ["G", "I"],
      },
      {
        id: "reports",
        label: "Reports",
        icon: <Glyph d={icons.chart} />,
        keys: ["G", "R"],
      },
      {
        id: "settings",
        label: "Settings",
        icon: <Glyph d={icons.gear} />,
        keys: ["G", "S"],
      },
      {
        id: "help",
        label: "Help & feedback",
        icon: <Glyph d={icons.help} />,
        keys: ["?"],
      },
    ],
  },
];

/** Lowercase and without accents, so "munchen" finds "München" */
const fold = (s: string) =>
  s.normalize("NFD").replace(/\p{M}/gu, "").toLowerCase();
/** First whole letter or digit – no half emoji */
const initial = (s: string) =>
  (s.match(/[\p{L}\p{N}]/u)?.[0] ?? "").toUpperCase();

/** The typed text, bold where it matches */
function Highlight({ text, query }: { text: string; query: string }) {
  const folded = fold(text);
  const needle = fold(query);
  // Only mark when folding kept every character in place – otherwise the mark would land on the wrong letters
  const i =
    needle && folded.length === text.length ? folded.indexOf(needle) : -1;
  if (i < 0) return <>{text}</>;
  return (
    <>
      {text.slice(0, i)}
      <mark>{text.slice(i, i + needle.length)}</mark>
      {text.slice(i + needle.length)}
    </>
  );
}

/** A command with a key of its own, so duplicate or odd ids never collide */
type Row = Command & { key: string; n: number };

/** A broken image logo falls back to the command's first letter */
const onBroken = (e: SyntheticEvent<HTMLElement>) => {
  e.currentTarget.dataset.broken = "";
};

type ItemProps = {
  cmd: Row;
  domId: string;
  selected: boolean;
  ran: boolean;
  mod: string;
  q: string;
  on: { current: { move: (cmd: Row) => void; run: (cmd: Row) => void } };
};

/** One row – memoized, so moving the highlight through thousands of commands only redraws two rows */
const Item = memo(function Item({
  cmd,
  domId,
  selected,
  ran,
  mod,
  q,
  on,
}: ItemProps) {
  const keys = cmd.n ? [mod, String(cmd.n)] : cmd.keys;
  return (
    <button
      id={domId}
      data-key={cmd.key}
      type="button"
      role="option"
      aria-selected={selected}
      data-ran={ran || undefined}
      className="ui-cmc__item"
      tabIndex={-1}
      onMouseMove={() => on.current.move(cmd)}
      onClick={() => on.current.run(cmd)}
    >
      {cmd.logo ? (
        <span
          className="ui-cmc__logo"
          data-letter={initial(cmd.label)}
          onErrorCapture={onBroken}
        >
          {cmd.logo}
        </span>
      ) : (
        <span
          className="ui-cmc__icon"
          data-letter={initial(cmd.label)}
          onErrorCapture={onBroken}
        >
          {cmd.icon}
        </span>
      )}
      <span className="ui-cmc__text">
        <span className="ui-cmc__name" dir="auto" title={cmd.label}>
          <Highlight text={cmd.label} query={q} />
        </span>
        {cmd.hint && (
          <span className="ui-cmc__hint" dir="auto" title={cmd.hint}>
            <Highlight text={cmd.hint} query={q} />
          </span>
        )}
      </span>
      {keys && (
        <span className="ui-cmc__keys" aria-hidden="true">
          {keys.map((k, i) => (
            <kbd key={i}>{k}</kbd>
          ))}
        </span>
      )}
    </button>
  );
});

const isMac = () =>
  typeof navigator !== "undefined" &&
  /Mac|iPhone|iPad/.test(navigator.platform);

export function CommandMenu({
  groups = defaultGroups,
  placeholder = "Type a command or search…",
  onRun,
  onClose,
  autoFocus,
  className,
}: CommandMenuProps) {
  const id = "ui-cmc" + useId().replace(/[^a-zA-Z0-9_-]/g, "");
  const input = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const innerRef = useRef<HTMLDivElement>(null);
  const [listHeight, setListHeight] = useState(0);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState<string | null>(null);
  const [ran, setRan] = useState<{ id: string; label: string } | null>(null);
  const [pill, setPill] = useState<{ y: number; h: number } | null>(null);
  // The highlight shows while the pointer is over the list or the keyboard moves – leaving the list fades it out
  const [lit, setLit] = useState(true);
  const mod = isMac() ? "⌘" : "Ctrl";

  // Every row gets a key of its own and a search text, once per groups – missing labels or commands from an API never crash
  const all = useMemo(
    () =>
      (groups ?? []).filter(Boolean).map((g, gi) => ({
        key: `g${gi}`,
        id: g.id,
        label: String(g.label ?? ""),
        commands: (Array.isArray(g.commands) ? g.commands : [])
          .filter(Boolean)
          .map((c, ci) => {
            const label = String(c.label ?? "");
            const hint =
              c.hint == null || c.hint === "" ? undefined : String(c.hint);
            const row: Row = { ...c, label, hint, key: `${gi}-${ci}`, n: 0 };
            return {
              row,
              text: fold(`${label} ${hint ?? ""} ${c.keywords ?? ""}`),
            };
          }),
      })),
    [groups],
  );
  const q = query.trim();
  const shown = useMemo(() => {
    const needle = fold(q);
    const result = all
      .map((g) => ({
        ...g,
        commands: g.commands
          .filter((c) => !needle || c.text.includes(needle))
          // ⌘1–⌘5 go to the first five suggestions shown – none beyond, since ⌘6 does nothing
          .map((c, i) =>
            g.id === "suggestions" && i < 5 ? { ...c.row, n: i + 1 } : c.row,
          ),
      }))
      .filter((g) => g.commands.length);
    // With text typed, the first row always offers to ask the assistant about it
    if (needle)
      result.unshift({
        key: "ask",
        id: "ask-about",
        label: "Assistant",
        commands: [
          {
            key: "ask",
            n: 0,
            id: "ask-about",
            label: `Ask assistant about “${q}”`,
            logo: <SparkTile />,
          },
        ],
      });
    return result;
  }, [all, q]);
  const order = useMemo(() => shown.flatMap((g) => g.commands), [shown]);
  const suggestions = order.filter((c) => c.n);
  const current = order.find((c) => c.key === active) ?? order[0];

  // Keyboard moves keep the active row in view; the mouse never scrolls the list
  const byKeyboard = useRef(false);

  // The highlight glides to the active row
  useLayoutEffect(() => {
    const list = listRef.current;
    const row = list?.querySelector<HTMLElement>(
      `[data-key="${current?.key}"]`,
    );
    if (!list || !row) return setPill(null);
    setPill({ y: row.offsetTop, h: row.offsetHeight });
    if (!byKeyboard.current) return;
    byKeyboard.current = false;
    // Scroll only the list (never the page), keeping a little room and the group title above the first row in view
    const pad = 8;
    const label = row.previousElementSibling?.classList.contains(
      "ui-cmc__label",
    )
      ? (row.previousElementSibling as HTMLElement).offsetHeight
      : 0;
    const top = row.offsetTop - pad - label;
    const bottom = row.offsetTop + row.offsetHeight + pad;
    if (top < list.scrollTop) list.scrollTop = top;
    else if (bottom > list.scrollTop + list.clientHeight)
      list.scrollTop = bottom - list.clientHeight;
  }, [current?.key, order.length]);

  // The list's height follows its rows (up to its max height), so searching makes the menu glide smaller, not jump
  useLayoutEffect(() => {
    const list = listRef.current;
    const inner = innerRef.current;
    if (!list || !inner) return;
    const measure = () => {
      const style = getComputedStyle(list);
      const pad =
        parseFloat(style.paddingTop) + parseFloat(style.paddingBottom);
      const max = parseFloat(style.maxHeight) || Infinity;
      setListHeight(Math.min(inner.offsetHeight + pad, max));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(inner);
    return () => ro.disconnect();
  }, []);

  // ⌘K / Ctrl K focuses the field from anywhere
  useEffect(() => {
    const onKey = (e: globalThis.KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        input.current?.focus();
        input.current?.select();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    if (autoFocus) input.current?.focus();
  }, [autoFocus]);

  // A short confirmation in the last row after running a command
  useEffect(() => {
    if (!ran) return;
    const t = window.setTimeout(() => setRan(null), 1800);
    return () => window.clearTimeout(t);
  }, [ran]);

  const run = (cmd: Row) => {
    setActive(cmd.key);
    setRan({
      id: cmd.key,
      label: cmd.key === "ask" ? `Asked the assistant about “${q}”` : cmd.label,
    });
    if (cmd.key === "ask") onRun?.("ask", q);
    else onRun?.(cmd.id);
  };

  // The rows call the latest handlers through a ref, so they don't redraw when the handlers change
  const on = useRef({ move: (_: Row) => {}, run: (_: Row) => {} });
  on.current = {
    move: (cmd) => {
      setLit(true);
      if (cmd.key !== current?.key) setActive(cmd.key);
    },
    run: (cmd) => {
      run(cmd);
      input.current?.focus({ preventScroll: true });
    },
  };

  const move = (step: number) => {
    if (!order.length || !current) return;
    byKeyboard.current = true;
    setLit(true);
    const i = order.findIndex((c) => c.key === current.key);
    setActive(order[(i + step + order.length) % order.length].key);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      move(e.key === "ArrowDown" ? 1 : -1);
    } else if (e.key === "Enter" && current) {
      e.preventDefault();
      run(current);
    } else if (e.key === "Tab" && shown.length > 1) {
      // Tab jumps to the first row of the next section, Shift Tab to the previous one
      e.preventDefault();
      const g = shown.findIndex((group) =>
        group.commands.some((c) => c.key === current?.key),
      );
      const next = e.shiftKey
        ? g <= 0
          ? shown.length - 1
          : g - 1
        : (g + 1) % shown.length;
      byKeyboard.current = true;
      setLit(true);
      setActive(shown[next].commands[0].key);
    } else if (e.key === "Escape") {
      if (query) setQuery("");
      else onClose?.();
    } else if ((e.metaKey || e.ctrlKey) && /^[1-5]$/.test(e.key)) {
      const cmd = suggestions[Number(e.key) - 1];
      if (cmd) {
        e.preventDefault();
        run(cmd);
      }
    }
  };

  return (
    <div className={["ui-cmc", className].filter(Boolean).join(" ")}>
      {/* React 19 hoists this into <head> once, no matter how many instances render */}
      <style href="ui-command-menu-clean" precedence="default">
        {css}
      </style>

      <div className="ui-cmc__card">
        <label className="ui-cmc__search">
          <svg viewBox="0 0 16 16" aria-hidden="true">
            <circle cx="7.25" cy="7.25" r="4.75" />
            <path d="m13.5 13.5-2.9-2.9" />
          </svg>
          <input
            ref={input}
            value={query}
            dir="auto"
            placeholder={placeholder}
            aria-label="Command"
            role="combobox"
            aria-expanded="true"
            aria-controls={`${id}-list`}
            aria-activedescendant={current ? `${id}-${current.key}` : undefined}
            onChange={(e) => {
              setQuery(e.target.value);
              setActive(null);
            }}
            onKeyDown={onKeyDown}
          />
          <kbd className="ui-cmc__enter" aria-hidden="true">
            <svg viewBox="0 0 16 16">
              <path d="M12.5 4v3a2 2 0 0 1-2 2h-7M6 6.5 3.5 9 6 11.5" />
            </svg>
          </kbd>
        </label>

        <div
          ref={listRef}
          id={`${id}-list`}
          className="ui-cmc__list"
          role="listbox"
          aria-label="Commands"
          style={listHeight ? { height: listHeight } : undefined}
          onMouseLeave={() => setLit(false)}
        >
          {pill && (
            <span
              className="ui-cmc__pill"
              data-lit={lit || undefined}
              style={{ transform: `translateY(${pill.y}px)`, height: pill.h }}
              aria-hidden="true"
            />
          )}
          <div ref={innerRef} className="ui-cmc__inner">
            {shown.map((group) => (
              <div
                key={group.key}
                role="group"
                aria-label={group.label}
                className="ui-cmc__group"
              >
                {group.key !== "ask" && group.label && (
                  <span className="ui-cmc__label" dir="auto">
                    {group.label}
                  </span>
                )}
                {group.commands.map((cmd) => (
                  <Item
                    key={cmd.key}
                    cmd={cmd}
                    domId={`${id}-${cmd.key}`}
                    selected={cmd.key === current?.key}
                    ran={ran?.id === cmd.key}
                    mod={mod}
                    q={q}
                    on={on}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>

        {/* The last row of the panel, under a hairline: a hint, or what just ran */}
        <footer className="ui-cmc__foot" aria-live="polite">
          {ran ? (
            <span className="ui-cmc__done" key={ran.label}>
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="m3.5 8.5 3 3 6-7" />
              </svg>
              <span className="ui-cmc__done-text">{ran.label}</span>
            </span>
          ) : (
            <span className="ui-cmc__tip">
              <kbd>Tab</kbd> to jump between sections
            </span>
          )}
          <span className="ui-cmc__foot-right">
            <span className="ui-cmc__keys">
              <kbd>{mod}</kbd>
              <kbd>K</kbd>
            </span>
            to open
          </span>
        </footer>
      </div>
    </div>
  );
}

/* ---------- Styles ---------- */

const css = /* css */ `
/* Colors and motion as variables – override them on .ui-cmc */
.ui-cmc {
  --ui-cmc-card: var(--aui-overlay, #ffffff);
  --ui-cmc-ink: var(--aui-ink, rgba(0, 0, 0, 0.88));
  --ui-cmc-strong: #000000;
  --ui-cmc-text: var(--aui-muted, rgba(0, 0, 0, 0.61));
  --ui-cmc-muted: var(--aui-muted, rgba(0, 0, 0, 0.45));
  --ui-cmc-faint: var(--aui-placeholder, rgba(0, 0, 0, 0.27));
  --ui-cmc-icon: var(--aui-muted, rgba(0, 0, 0, 0.447));
  --ui-cmc-line: var(--aui-border, rgba(0, 0, 0, 0.06));
  --ui-cmc-line-strong: var(--aui-border-strong, rgba(0, 0, 0, 0.1));
  --ui-cmc-active: var(--aui-subtle, rgba(0, 0, 0, 0.06));
  --ui-cmc-tile: #ffffff;
  --ui-cmc-tile-ring: rgba(0, 0, 0, 0.1);
  --ui-cmc-mark: rgba(0, 96, 226, 0.12);
  --ui-cmc-accent: rgba(0, 96, 226, 0.765);
  --ui-cmc-ok: var(--aui-success, #1e8a4f);
  /* Strokes are shadows: a 0.5px hairline plus a soft lift, like a popover */
  --ui-cmc-shadow: var(--aui-shadow-popover, 0 0 0 0.5px var(--aui-border, rgba(0, 0, 0, 0.08)), 0 12px 28px -10px rgba(0, 0, 0, 0.16), 0 1px 2px 0 rgba(0, 0, 0, 0.05));
  --ui-cmc-ease: var(--aui-ease, cubic-bezier(0.22, 1, 0.36, 1));
  box-sizing: border-box;
  container-type: inline-size;
  display: grid;
  /* One column that can shrink – a long confirmation never widens the menu */
  grid-template-columns: minmax(0, 1fr);
  width: 100%;
  max-width: 600px;
  font-family: var(--aui-font, "Inter", "InterVariable", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif);
  font-size: 13px;
  line-height: 20px;
  font-weight: 400;
  color: var(--ui-cmc-ink);
  -webkit-font-smoothing: antialiased;
}
.ui-cmc *, .ui-cmc *::before, .ui-cmc *::after { box-sizing: border-box; }
:where(.ui-cmc) button { margin: 0; padding: 0; border: 0; background: none; font: inherit; color: inherit; cursor: pointer; }
:where(.ui-cmc) svg { display: block; flex: none; }

/* One white panel: search, results and the last row – the edge is a shadow, not a border */
.ui-cmc__card { overflow: hidden; border-radius: calc(var(--aui-radius-lg, 10px) + 2px); background: var(--ui-cmc-card); box-shadow: var(--ui-cmc-shadow); }

/* Search row */
.ui-cmc__search { display: flex; align-items: center; gap: 12px; height: 48px; padding: 0 12px 0 14px; border-bottom: 0.5px solid var(--ui-cmc-line); cursor: text; }
.ui-cmc__search > svg { width: 16px; height: 16px; fill: none; stroke: var(--ui-cmc-icon); stroke-width: 1.1; stroke-linecap: round; }
.ui-cmc__search input { flex: 1; min-width: 0; height: 20px; padding: 0; border: 0; background: transparent; font: inherit; font-size: 13px; line-height: 20px; color: var(--ui-cmc-ink); caret-color: var(--ui-cmc-accent); outline: none; }
.ui-cmc__search input::placeholder { color: var(--ui-cmc-faint); }
.ui-cmc__search input:focus-visible { outline: none; }
.ui-cmc__enter svg { width: 12px; height: 12px; fill: none; stroke: currentColor; stroke-width: 1.1; stroke-linecap: round; stroke-linejoin: round; }

/* Results */
.ui-cmc__list { position: relative; max-height: 360px; padding: 0 4px 4px; overflow-y: auto; scrollbar-width: thin; transition: height 320ms var(--ui-cmc-ease); }
.ui-cmc__group { display: flex; flex-direction: column; }
.ui-cmc__label { padding: 10px 8px 4px; overflow-wrap: anywhere; font-size: 10px; font-weight: 500; line-height: 16px; letter-spacing: 0.5px; text-transform: uppercase; color: var(--ui-cmc-muted); }
/* With text typed, the assistant row sits on top without a label */
.ui-cmc__group:first-child:not(:has(.ui-cmc__label)) { padding-top: 4px; }
/* The highlight follows the active row */
.ui-cmc__pill {
  position: absolute;
  top: 0;
  left: 4px;
  right: 4px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 1px);
  background: var(--ui-cmc-active);
  opacity: 0;
  /* A short, crisp move between rows – no long glide; it fades out when the pointer leaves the list */
  transition: transform 90ms ease-out, height 90ms ease-out, opacity 140ms ease;
}
.ui-cmc__pill[data-lit] { opacity: 1; }
.ui-cmc__item { position: relative; display: flex; align-items: center; gap: 10px; height: 36px; padding: 0 8px; border-radius: calc(var(--aui-radius-sm, 5px) + 1px); text-align: start; }
.ui-cmc__logo {
  display: grid;
  place-items: center;
  flex: none;
  width: 20px;
  height: 20px;
  border-radius: 5px;
  background: var(--ui-cmc-tile);
  box-shadow: 0 0 0 0.5px var(--ui-cmc-tile-ring), 0 1px 1px -1px rgba(0, 0, 0, 0.1);
  color: var(--ui-cmc-strong);
}
.ui-cmc__logo svg { width: 12px; height: 12px; }
/* Image logos fit the tile; a broken one shows the command's first letter instead */
.ui-cmc__logo, .ui-cmc__icon { overflow: hidden; }
.ui-cmc__logo img { width: 12px; height: 12px; object-fit: contain; }
.ui-cmc__icon img { width: 16px; height: 16px; object-fit: contain; }
.ui-cmc__logo[data-broken] > *, .ui-cmc__icon[data-broken] > * { display: none; }
.ui-cmc__logo[data-broken]::after, .ui-cmc__icon[data-broken]::after { content: attr(data-letter); font-size: 10px; font-weight: 600; line-height: 1; color: var(--ui-cmc-text); }
.ui-cmc__icon { display: grid; place-items: center; flex: none; width: 20px; height: 20px; color: var(--ui-cmc-icon); transition: color 140ms ease; }
.ui-cmc__glyph { width: 16px; height: 16px; fill: none; stroke: currentColor; stroke-width: 1.1; stroke-linecap: round; stroke-linejoin: round; }
.ui-cmc__glyph path { vector-effect: non-scaling-stroke; }
.ui-cmc__item[aria-selected="true"] .ui-cmc__icon { color: var(--ui-cmc-ink); }
.ui-cmc__text { display: flex; flex: 1; align-items: baseline; gap: 8px; min-width: 0; }
.ui-cmc__name { overflow: hidden; color: var(--ui-cmc-ink); white-space: nowrap; text-overflow: ellipsis; }
.ui-cmc__hint { overflow: hidden; color: var(--ui-cmc-muted); white-space: nowrap; text-overflow: ellipsis; }
.ui-cmc mark { border-radius: 3px; background: var(--ui-cmc-mark); color: inherit; }

/* Keyboard hints: small chips with an inset hairline */
.ui-cmc__keys { display: inline-flex; flex: none; gap: 2px; }
.ui-cmc kbd {
  display: inline-grid;
  place-items: center;
  min-width: 20px;
  height: 20px;
  padding: 0 5px;
  border-radius: var(--aui-radius-sm, 5px);
  box-shadow: inset 0 0 0 0.5px var(--ui-cmc-line-strong);
  font-family: inherit;
  font-size: 11px;
  font-weight: 500;
  line-height: 16px;
  color: var(--ui-cmc-muted);
  transition: color 140ms ease;
}
.ui-cmc__item[aria-selected="true"] kbd { color: var(--ui-cmc-text); }
.ui-cmc__enter { padding: 0 4px; color: var(--ui-cmc-icon); }
/* A short press on the row that just ran */
.ui-cmc__item[data-ran] { animation: ui-cmc-ran 600ms var(--ui-cmc-ease); }
@keyframes ui-cmc-ran { 30% { transform: scale(0.985); } }

/* The last row of the panel */
.ui-cmc__foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  height: 40px;
  padding: 0 12px;
  border-top: 0.5px solid var(--ui-cmc-line);
  font-size: 12px;
  line-height: 16px;
  color: var(--ui-cmc-muted);
}
.ui-cmc__foot > span { display: inline-flex; align-items: center; gap: 6px; min-width: 0; white-space: nowrap; }
.ui-cmc__done { color: var(--ui-cmc-ink); animation: ui-cmc-in 260ms var(--ui-cmc-ease); }
.ui-cmc__done-text { overflow: hidden; text-overflow: ellipsis; }
.ui-cmc__done svg { width: 16px; height: 16px; fill: none; stroke: var(--ui-cmc-ok); stroke-width: 1.4; stroke-linecap: round; stroke-linejoin: round; }
.ui-cmc__foot-right { flex: none; }
@keyframes ui-cmc-in { from { opacity: 0; transform: translateY(3px); } to { opacity: 1; transform: none; } }

/* Narrow (phones): no shortcut keys or hints – there's no keyboard to use them with */
@container (max-width: 480px) {
  .ui-cmc__item .ui-cmc__keys, .ui-cmc__foot > .ui-cmc__foot-right, .ui-cmc__foot > .ui-cmc__tip, .ui-cmc__enter, .ui-cmc__hint { display: none; }
  .ui-cmc__item { height: 40px; }
  .ui-cmc__list { max-height: 360px; }
  .ui-cmc__foot:not(:has(.ui-cmc__done)) { display: none; }
}
/* Touch screens: taller rows */
@media (hover: none) and (pointer: coarse) {
  .ui-cmc__item { height: 44px; }
  .ui-cmc__search { height: 52px; }
}
@media (prefers-reduced-motion: reduce) {
  .ui-cmc *, .ui-cmc *::before, .ui-cmc *::after { animation: none !important; transition: none !important; }
}

/* Dark mode: follows a .dark class or data-theme="dark" on any ancestor, e.g. <html> */
:where(.dark, [data-theme="dark"]) .ui-cmc {
  --ui-cmc-card: var(--aui-overlay, #1c1c1e);
  --ui-cmc-ink: var(--aui-ink, rgba(255, 255, 255, 0.9));
  --ui-cmc-strong: #ffffff;
  --ui-cmc-text: var(--aui-muted, rgba(255, 255, 255, 0.62));
  --ui-cmc-muted: var(--aui-muted, rgba(255, 255, 255, 0.46));
  --ui-cmc-faint: var(--aui-placeholder, rgba(255, 255, 255, 0.3));
  --ui-cmc-icon: var(--aui-muted, rgba(255, 255, 255, 0.5));
  --ui-cmc-line: var(--aui-border, rgba(255, 255, 255, 0.08));
  --ui-cmc-line-strong: var(--aui-border-strong, rgba(255, 255, 255, 0.12));
  --ui-cmc-active: var(--aui-subtle, rgba(255, 255, 255, 0.08));
  --ui-cmc-tile: #2a2a2d;
  --ui-cmc-tile-ring: rgba(255, 255, 255, 0.12);
  --ui-cmc-mark: rgba(77, 141, 255, 0.24);
  --ui-cmc-accent: var(--aui-blue, #4fb0fe);
  --ui-cmc-ok: var(--aui-green, #5fc88a);
  --ui-cmc-shadow: var(--aui-shadow-popover, 0 0 0 0.5px var(--aui-border, rgba(255, 255, 255, 0.1)), 0 1px 2px 0 rgba(0, 0, 0, 0.4));
}
`;
