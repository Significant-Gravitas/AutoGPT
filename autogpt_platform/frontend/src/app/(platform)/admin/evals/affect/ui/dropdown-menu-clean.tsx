"use client";

import {
  createElement,
  Fragment,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type FocusEvent as ReactFocusEvent,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from "react";

/*
 * Dropdown Menu – a single self-contained file, clean style.
 * A menu of actions or choices that opens from any trigger: icons, a quiet second line, shortcuts, dividers, red
 * items for what can't be undone, checks for choices (radio or checkbox) and submenus that open to the side. One
 * soft highlight glides to the row under the pointer or the arrow keys; typing a letter jumps to the item. The menu
 * sits in the top layer, so no overflow cuts it, and it flips to whichever side has room.
 * The styles are injected below, so no extra files are needed.
 */

/** An icon from Hugeicons (e.g. `import { Copy01Icon } from "@hugeicons/core-free-icons"`) – drawn inline */
export type HugeIcon = readonly (readonly [
  string,
  { readonly [key: string]: string | number },
])[];

export type DropdownMenuItem = {
  label: string;
  /** A second line in a lighter color */
  description?: string;
  /** A 16px glyph before the label: an inline SVG or a Hugeicons icon object */
  icon?: ReactNode | HugeIcon;
  /** A brand mark (Gmail, Slack …) in place of the icon, in its own colors */
  logo?: ReactNode;
  /** Shows a check – the item becomes a choice (true or false; leave it out for a plain action) */
  checked?: boolean;
  /** How a choice is announced: one of a set (radio, the default) or on/off on its own (checkbox) */
  type?: "radio" | "checkbox";
  /** Red text and icon, e.g. Delete */
  danger?: boolean;
  /** A keyboard shortcut on the right, e.g. "⌥⌘R" – shown only, wire the keys up yourself */
  shortcut?: string;
  /** A line above this item, to group the ones after it */
  divider?: boolean;
  /** Shown, but can't be picked; the arrow keys skip it */
  disabled?: boolean;
  /** The menu stays open after a pick – for checkboxes you tick several of */
  keepOpen?: boolean;
  /** More items that open to the side on hover, →, Enter or Space */
  submenu?: DropdownMenuItem[];
  onSelect?: () => void;
};

/** Spread these on the trigger: they wire up its name, state and the click */
export type DropdownMenuTriggerProps = {
  id: string;
  "aria-haspopup": "menu";
  "aria-expanded": boolean;
  "aria-controls"?: string;
  onClick: (event?: { detail?: number }) => void;
};

export type DropdownMenuProps = {
  /** Draws the trigger – it gets the open state, a toggle and the props to spread on it */
  trigger: (props: {
    open: boolean;
    toggle: (event?: { detail?: number }) => void;
    triggerProps: DropdownMenuTriggerProps;
  }) => ReactNode;
  items: DropdownMenuItem[];
  /** start: the menu's left edge on the trigger's · end: the right edges line up */
  align?: "start" | "end";
  /** Where it opens; it flips when there is no room */
  side?: "bottom" | "top";
  /** A small heading above the items – it also names the menu */
  title?: string;
  /** Anything above the items, e.g. who is signed in */
  header?: ReactNode;
  /** Makes it controlled */
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  /** Classes on the menu panel itself (className goes on the wrapper around trigger and menu) */
  menuClassName?: string;
  /** The menu's width – by default it fits its items (at least 168px) */
  width?: number | string;
  /** Shown when there are no items */
  empty?: ReactNode;
  /** On the wrapper around the trigger */
  className?: string;
  /** Names the menu when there is no title */
  "aria-label"?: string;
};

type Start = "first" | "last" | "menu";

export function DropdownMenu({
  trigger,
  items,
  align = "start",
  side = "bottom",
  title,
  header,
  open: openProp,
  defaultOpen = false,
  onOpenChange,
  menuClassName,
  width,
  empty = "Nothing here yet",
  className,
  "aria-label": ariaLabel,
}: DropdownMenuProps) {
  const uid = useId();
  const triggerId = `${uid}-trigger`;
  const menuId = `${uid}-menu`;
  const titleId = `${uid}-title`;

  const [own, setOwn] = useState(defaultOpen);
  const open = openProp ?? own;
  const anchor = useRef<HTMLSpanElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  // Where the focus goes when it opens; open from the start (a demo, a restored view) leaves the focus alone
  const start = useRef<Start>("first");
  const wasClosed = useRef(!open);
  if (!open) wasClosed.current = true;
  const label = useRef<string | undefined>(triggerId);
  const list = Array.isArray(items) ? items.filter(Boolean) : [];

  const setOpen = (next: boolean) => {
    if (next === open) return;
    if (openProp === undefined) setOwn(next);
    onOpenChange?.(next);
  };
  const trig = () => triggerOf(anchor.current, panel.current);
  /** focusBack: after the pick (unless the pick moved the focus) · "now": Escape and Tab – Tab then moves on from the trigger */
  const close = (focusBack: boolean | "now") => {
    if (focusBack === "now") trig()?.focus({ preventScroll: true });
    setOpen(false);
    if (focusBack === true) refocus(trig());
  };
  const toggle = (event?: { detail?: number }) => {
    // A pointer click opens it with the focus on the menu itself; Enter, Space or a call from code on the first item
    start.current = event?.detail ? "menu" : "first";
    setOpen(!open);
  };

  // A trigger that doesn't spread triggerProps still gets a name to point at and says it opens a menu
  useEffect(() => {
    const t = trig();
    if (!t) return;
    if (!t.id) t.id = triggerId;
    if (!t.hasAttribute("aria-haspopup"))
      t.setAttribute("aria-haspopup", "menu");
    label.current = t.id;
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Into the top layer and next to the trigger, before paint; it follows on scroll and resize
  useLayoutEffect(() => {
    const el = panel.current;
    if (!open || !el) return;
    showLayer(el);
    const update = () =>
      anchor.current &&
      place(el, anchor.current.getBoundingClientRect(), side, align, 6);
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    const ro =
      typeof ResizeObserver === "undefined" ? null : new ResizeObserver(update);
    ro?.observe(el);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
      ro?.disconnect();
      hideLayer(el);
    };
  }, [open, side, align]);

  // The focus moves in: onto the checked item (or the first, or the last after ↑), or onto the menu after a click
  useEffect(() => {
    if (!open) return;
    const menu = panel.current?.querySelector<HTMLElement>(
      ":scope > .ui-dmc__list",
    );
    const how = start.current;
    if (!menu || !wasClosed.current) return;
    if (how === "menu") return menu.focus({ preventScroll: true });
    const rows = itemsOf(menu);
    const checked = rows.find((r) => r.getAttribute("aria-checked") === "true");
    focusRow(
      (how === "last" ? rows[rows.length - 1] : (checked ?? rows[0])) ?? menu,
    );
  }, [open]);

  // A press anywhere else closes it, without taking the focus back
  useEffect(() => {
    if (!open) return;
    const down = (e: PointerEvent) => {
      if (!anchor.current?.contains(e.target as Node)) close(false);
    };
    document.addEventListener("pointerdown", down, true);
    return () => document.removeEventListener("pointerdown", down, true);
  }); // eslint-disable-line react-hooks/exhaustive-deps

  // ↓ or ↑ on the closed trigger opens it, like Enter and Space do
  const onKeyDown = (e: ReactKeyboardEvent<HTMLSpanElement>) => {
    if (
      open ||
      (e.key !== "ArrowDown" && e.key !== "ArrowUp") ||
      panel.current?.contains(e.target as Node)
    )
      return;
    e.preventDefault();
    start.current = e.key === "ArrowUp" ? "last" : "first";
    setOpen(true);
  };
  // Tabbing away from the trigger and its menu closes it
  const onBlur = (e: ReactFocusEvent<HTMLSpanElement>) => {
    const to = e.relatedTarget as Node | null;
    if (open && to && !anchor.current?.contains(to)) close(false);
  };

  const triggerProps: DropdownMenuTriggerProps = {
    id: triggerId,
    "aria-haspopup": "menu",
    "aria-expanded": open,
    "aria-controls": open ? menuId : undefined,
    onClick: toggle,
  };

  return (
    <span
      ref={anchor}
      className={["ui-dmc", className].filter(Boolean).join(" ")}
      onKeyDown={onKeyDown}
      onBlur={onBlur}
    >
      {/* React 19 hoists this into <head> once, no matter how many menus render */}
      <style href="ui-dropdown-menu-clean" precedence="default">
        {css}
      </style>
      {trigger({ open, toggle, triggerProps })}
      {open && (
        <div
          ref={panel}
          className={["ui-dmc__panel", menuClassName].filter(Boolean).join(" ")}
          popover="manual"
          data-side={side}
          data-align={align}
          style={
            width != null
              ? { width, maxWidth: "calc(100vw - 16px)" }
              : undefined
          }
        >
          {header}
          {title && (
            <span className="ui-dmc__title" id={titleId}>
              {title}
            </span>
          )}
          <MenuLevel
            id={menuId}
            items={list}
            empty={empty}
            labelledBy={title ? titleId : ariaLabel ? undefined : label.current}
            label={title ? undefined : ariaLabel}
            onPick={() => close(true)}
            onClose={close}
          />
        </div>
      )}
    </span>
  );
}

/* ---------- One level of the menu: the root, or a submenu ---------- */

type LevelProps = {
  id: string;
  items: DropdownMenuItem[];
  empty?: ReactNode;
  labelledBy?: string;
  label?: string;
  /** A pick that closes the whole menu */
  onPick: () => void;
  /** Escape at the root, Tab anywhere – the trigger takes the focus */
  onClose: (focusBack: boolean | "now") => void;
  /** ← or Escape in a submenu: back to the item that opened it */
  onBack?: () => void;
};

function MenuLevel({
  id,
  items,
  empty,
  labelledBy,
  label,
  onPick,
  onClose,
  onBack,
}: LevelProps) {
  const uid = useId();
  const list = useRef<HTMLDivElement>(null);
  const [glow, setGlow] = useState<{
    top: number;
    height: number;
    danger: boolean;
  } | null>(null);
  const [sub, setSub] = useState<{ index: number; focus: boolean } | null>(
    null,
  );
  const timer = useRef(0);
  /** What the timer will do: open the submenu of this index, or -1 to close the open one */
  const pending = useRef<number | null>(null);
  const typed = useRef({ text: "", at: 0 });

  useEffect(() => () => window.clearTimeout(timer.current), []);
  const stopTimer = () => {
    window.clearTimeout(timer.current);
    pending.current = null;
  };
  const later = (fn: () => void, ms: number, what: number) => {
    window.clearTimeout(timer.current);
    pending.current = what;
    timer.current = window.setTimeout(() => {
      pending.current = null;
      fn();
    }, ms);
  };

  const rowAt = (i: number) =>
    list.current?.querySelector<HTMLElement>(`:scope > [data-index="${i}"]`) ??
    null;
  const openSub = (index: number, focus: boolean) => {
    stopTimer();
    setSub((s) =>
      s?.index === index ? (focus ? { index, focus } : s) : { index, focus },
    );
    // Already open from a hover: the keyboard steps into it right away
    if (focus && sub?.index === index)
      focusRow(
        itemsOf(
          document
            .getElementById(`${uid}-sub`)
            ?.querySelector(".ui-dmc__list") ?? null,
        )[0],
      );
  };
  const back = () => {
    const at = sub?.index;
    setSub(null);
    if (at != null) focusRow(rowAt(at));
  };

  const activate = (i: number, keyboard: boolean) => {
    const item = items[i];
    if (!item || item.disabled) return;
    if (item.submenu) {
      if (!keyboard && sub?.index === i && !sub.focus) setSub(null);
      else openSub(i, keyboard);
      return;
    }
    item.onSelect?.();
    if (!item.keepOpen) onPick();
  };

  // Typing letters jumps to the next item that starts with them
  const typeahead = (key: string) => {
    const now = Date.now();
    typed.current.text =
      now - typed.current.at > 700 ? key : typed.current.text + key;
    typed.current.at = now;
    const q = typed.current.text.toLowerCase();
    const rows = itemsOf(list.current);
    const at = rows.indexOf(document.activeElement as HTMLElement);
    const from = q.length === 1 ? at + 1 : Math.max(at, 0);
    for (let k = 0; k < rows.length; k++) {
      const row = rows[(from + k) % rows.length];
      if (
        (items[Number(row.dataset.index)]?.label ?? "")
          .toLowerCase()
          .startsWith(q)
      )
        return focusRow(row);
    }
  };

  const onKeyDown = (e: ReactKeyboardEvent<HTMLDivElement>) => {
    const rows = itemsOf(list.current);
    const at = rows.indexOf(document.activeElement as HTMLElement);
    const index = at >= 0 ? Number(rows[at].dataset.index) : -1;
    const item = index >= 0 ? items[index] : undefined;
    const stop = () => {
      e.preventDefault();
      e.stopPropagation();
    };
    const go = (i: number) => {
      stop();
      // Moving on from an item closes a submenu the pointer had opened
      if (sub) setSub(null);
      if (rows.length) focusRow(rows[(i + rows.length) % rows.length]);
    };
    switch (e.key) {
      case "ArrowDown":
        return go(at + 1);
      case "ArrowUp":
        return go(at < 0 ? -1 : at - 1);
      case "Home":
      case "PageUp":
        return go(0);
      case "End":
      case "PageDown":
        return go(-1);
      case "ArrowRight":
        if (item?.submenu) {
          stop();
          openSub(index, true);
        }
        return;
      case "ArrowLeft":
        if (onBack) {
          stop();
          onBack();
        }
        return;
      case "Escape":
        stop();
        return onBack ? onBack() : onClose("now");
      case "Enter":
      case " ":
        stop();
        if (e.key === " " && Date.now() - typed.current.at < 700)
          return typeahead(" ");
        return activate(index, true);
      case "Tab":
        // The focus goes to the trigger first, so Tab moves on from there
        return onClose("now");
      default:
        if (e.key.length === 1 && !e.metaKey && !e.ctrlKey && !e.altKey) {
          stop();
          typeahead(e.key);
        }
    }
  };

  // The highlight follows the focused row – the pointer focuses the row it is on
  const onFocus = (e: ReactFocusEvent<HTMLDivElement>) => {
    const row = e.target as HTMLElement;
    if (row === list.current || row.parentElement !== list.current)
      return setGlow(null);
    setGlow({
      top: row.offsetTop,
      height: row.offsetHeight,
      danger: row.hasAttribute("data-danger"),
    });
  };
  const onBlur = (e: ReactFocusEvent<HTMLDivElement>) => {
    const to = e.relatedTarget as HTMLElement | null;
    if (!to || to.parentElement !== list.current) setGlow(null);
  };

  const subItems = sub ? items[sub.index]?.submenu : undefined;

  return (
    <>
      <div
        ref={list}
        id={id}
        className="ui-dmc__list"
        role="menu"
        aria-orientation="vertical"
        aria-labelledby={labelledBy}
        aria-label={label}
        tabIndex={-1}
        onKeyDown={onKeyDown}
        onFocus={onFocus}
        onBlur={onBlur}
        onPointerLeave={() => {
          stopTimer();
          // The pointer left: the highlight goes, the keys start again from the menu
          if (
            list.current?.contains(document.activeElement) &&
            document.activeElement !== list.current
          )
            list.current?.focus({ preventScroll: true });
        }}
      >
        {glow && (
          <span
            className="ui-dmc__glow"
            aria-hidden="true"
            data-danger={glow.danger || undefined}
            style={{
              transform: `translateY(${glow.top}px)`,
              height: glow.height,
            }}
          />
        )}
        {items.length === 0 && <span className="ui-dmc__empty">{empty}</span>}
        {items.map((item, i) => {
          const choice = item.checked !== undefined && !item.submenu;
          const role = !choice
            ? "menuitem"
            : item.type === "checkbox"
              ? "menuitemcheckbox"
              : "menuitemradio";
          const isOpen = sub?.index === i;
          return (
            <Fragment key={`${i}-${item.label}`}>
              {item.divider && i > 0 && (
                <span className="ui-dmc__sep" role="separator" />
              )}
              <div
                className="ui-dmc__item"
                role={role}
                tabIndex={-1}
                data-index={i}
                aria-checked={choice ? Boolean(item.checked) : undefined}
                aria-disabled={item.disabled || undefined}
                aria-haspopup={item.submenu ? "menu" : undefined}
                aria-expanded={item.submenu ? isOpen : undefined}
                aria-controls={
                  item.submenu && isOpen ? `${uid}-sub-list` : undefined
                }
                data-danger={item.danger || undefined}
                data-rich={item.description ? "" : undefined}
                data-open={isOpen || undefined}
                onPointerDown={(e) => e.preventDefault()}
                onPointerMove={(e) => {
                  const row = e.currentTarget;
                  if (item.disabled) {
                    if (document.activeElement !== list.current)
                      list.current?.focus({ preventScroll: true });
                    return;
                  }
                  if (document.activeElement !== row)
                    row.focus({ preventScroll: true });
                  // Rest on an item with a submenu to open it; on another item the open one closes after a beat
                  if (item.submenu) {
                    if (isOpen) stopTimer();
                    else if (pending.current !== i)
                      later(() => openSub(i, false), 90, i);
                  } else if (!sub) stopTimer();
                  else if (pending.current !== -1)
                    later(() => setSub(null), 220, -1);
                }}
                onClick={(e) => activate(i, e.detail === 0)}
              >
                {(item.logo || item.icon) && (
                  <span
                    className="ui-dmc__icon"
                    data-logo={item.logo ? "" : undefined}
                    aria-hidden="true"
                  >
                    {item.logo ?? <Glyph icon={item.icon} />}
                  </span>
                )}
                <span className="ui-dmc__label">
                  <span className="ui-dmc__text" title={item.label}>
                    {item.label}
                  </span>
                  {item.description && (
                    <span className="ui-dmc__desc" title={item.description}>
                      {item.description}
                    </span>
                  )}
                </span>
                {item.shortcut && (
                  <kbd className="ui-dmc__kbd">{item.shortcut}</kbd>
                )}
                {item.submenu && (
                  <span className="ui-dmc__more" aria-hidden="true">
                    <ChevronIcon />
                  </span>
                )}
                {choice && (
                  <span className="ui-dmc__check" aria-hidden="true">
                    {item.checked && <CheckIcon />}
                  </span>
                )}
              </div>
            </Fragment>
          );
        })}
      </div>
      {sub && subItems && (
        <SubMenu
          id={`${uid}-sub`}
          listId={`${uid}-sub-list`}
          row={() => rowAt(sub.index)}
          items={subItems}
          focus={sub.focus}
          label={items[sub.index]?.label}
          onEnter={() => stopTimer()}
          onBack={back}
          onPick={onPick}
          onClose={onClose}
        />
      )}
    </>
  );
}

/** A submenu: its own small panel in the top layer, beside the item that opened it (or on the other side) */
function SubMenu({
  id,
  listId,
  row,
  items,
  focus,
  label,
  onEnter,
  onBack,
  onPick,
  onClose,
}: {
  id: string;
  listId: string;
  row: () => HTMLElement | null;
  items: DropdownMenuItem[];
  focus: boolean;
  label?: string;
  onEnter: () => void;
  onBack: () => void;
  onPick: () => void;
  onClose: (focusBack: boolean | "now") => void;
}) {
  const panel = useRef<HTMLDivElement>(null);

  useLayoutEffect(() => {
    const el = panel.current;
    if (!el) return;
    showLayer(el);
    const update = () => {
      const r = row()?.getBoundingClientRect();
      // Its first item lines up with the item that opened it: the panel's 4px padding sits above it
      if (r)
        place(
          el,
          new DOMRect(r.left - 4, r.top - 4, r.width + 8, r.height + 8),
          "right",
          "start",
          4,
        );
    };
    update();
    window.addEventListener("resize", update);
    window.addEventListener("scroll", update, true);
    return () => {
      window.removeEventListener("resize", update);
      window.removeEventListener("scroll", update, true);
      hideLayer(el);
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (focus)
      focusRow(
        itemsOf(panel.current?.querySelector(".ui-dmc__list") ?? null)[0],
      );
  }, [focus]);

  return (
    <div
      ref={panel}
      id={id}
      className="ui-dmc__panel"
      data-sub=""
      popover="manual"
      onPointerEnter={onEnter}
    >
      <MenuLevel
        id={listId}
        items={items}
        label={label}
        onPick={onPick}
        onClose={onClose}
        onBack={onBack}
      />
    </div>
  );
}

/* ---------- Helpers ---------- */

/** The rows of one level that can be focused – not separators, not disabled ones, not a submenu's */
const itemsOf = (menu: Element | null) =>
  menu
    ? [
        ...menu.querySelectorAll<HTMLElement>(
          ':scope > [role^="menuitem"]:not([aria-disabled="true"])',
        ),
      ]
    : [];

function focusRow(row: HTMLElement | null | undefined) {
  if (!row) return;
  row.focus({ preventScroll: true });
  row.scrollIntoView?.({ block: "nearest" });
}

/** The trigger: the anchor's first control that isn't inside the menu */
function triggerOf(anchor: HTMLElement | null, panel: HTMLElement | null) {
  if (!anchor) return null;
  const all = anchor.querySelectorAll<HTMLElement>(
    "button, [href], input, [tabindex]",
  );
  for (const el of all) if (!panel?.contains(el)) return el;
  return null;
}

/** After a pick the menu is gone – the trigger takes the focus back, unless the pick moved it somewhere (a rename field) */
const refocus = (trigger: HTMLElement | null) =>
  window.setTimeout(() => {
    const at = document.activeElement;
    if (trigger?.isConnected && (!at || at === document.body))
      trigger.focus({ preventScroll: true });
  });

/** The top layer, so no overflow cuts it and no z-index fights it */
function showLayer(el: HTMLElement) {
  if ("showPopover" in el && !el.matches(":popover-open")) el.showPopover();
}
function hideLayer(el: HTMLElement) {
  if ("hidePopover" in el && el.isConnected && el.matches(":popover-open"))
    el.hidePopover();
}

/**
 * Puts a fixed panel next to a rectangle: on the preferred side, or the other one when that has more room; the
 * edge it lines up with switches when it would stick out, and it is pushed inside the window as a last resort.
 * Too tall for the room it gets: it scrolls.
 */
function place(
  el: HTMLElement,
  r: DOMRect,
  side: "top" | "bottom" | "left" | "right",
  align: "start" | "end",
  gap: number,
) {
  const pad = 8;
  // The trigger's width, for a menu that should be at least as wide: min-width: var(--ui-dmc-trigger-w)
  el.style.setProperty("--ui-dmc-trigger-w", `${Math.round(r.width)}px`);
  const vw = document.documentElement.clientWidth || window.innerWidth;
  const vh = window.innerHeight;
  el.style.maxHeight = "";
  const w = el.offsetWidth;
  const h = el.offsetHeight;
  const room = {
    bottom: vh - r.bottom - gap - pad,
    top: r.top - gap - pad,
    right: vw - r.right - gap - pad,
    left: r.left - gap - pad,
  };
  const flip = {
    top: "bottom",
    bottom: "top",
    left: "right",
    right: "left",
  } as const;
  let s: keyof typeof room = side;
  let a = align;
  let top: number;
  let left: number;
  const vertical = s === "top" || s === "bottom";
  if (room[s] < (vertical ? h : w) && room[flip[s]] > room[s]) s = flip[s];
  if (!vertical && room[s] < w) {
    // No room beside it on a narrow screen: under the row, a little indented
    s = "bottom";
    top = r.bottom - 2;
    left = Math.min(Math.max(pad, r.left + 16), vw - pad - w);
    el.style.maxHeight = `${Math.max(120, vh - top - pad)}px`;
  } else if (s === "top" || s === "bottom") {
    const fit = Math.min(h, Math.max(120, room[s]));
    top = s === "bottom" ? r.bottom + gap : r.top - gap - fit;
    left = a === "start" ? r.left : r.right - w;
    if (a === "start" && left + w > vw - pad && r.right - w >= pad)
      ((a = "end"), (left = r.right - w));
    else if (a === "end" && left < pad && r.left + w <= vw - pad)
      ((a = "start"), (left = r.left));
    left = Math.min(Math.max(pad, left), vw - pad - w);
    if (h > fit) el.style.maxHeight = `${fit}px`;
  } else {
    left = s === "right" ? r.right + gap : r.left - gap - w;
    top = a === "start" ? r.top : r.bottom - h;
    if (top + h > vh - pad) top = Math.max(pad, vh - pad - h);
    if (top < pad) top = pad;
    if (h > vh - 2 * pad) el.style.maxHeight = `${vh - 2 * pad}px`;
  }
  el.style.top = `${Math.round(top)}px`;
  el.style.left = `${Math.round(left)}px`;
  el.dataset.side = s;
  el.dataset.align = a;
}

const isHugeIcon = (icon: unknown): icon is HugeIcon =>
  Array.isArray(icon) &&
  icon.length > 0 &&
  icon.every((part) => Array.isArray(part) && typeof part[0] === "string");

/** Draws a Hugeicons icon object as an inline SVG; anything else is shown as it is */
function Glyph({ icon }: { icon: ReactNode | HugeIcon }) {
  if (!isHugeIcon(icon)) return <>{icon}</>;
  return (
    <svg viewBox="0 0 24 24" fill="none" focusable="false">
      {icon.map(([tag, { key, ...attrs }], i) =>
        createElement(tag, { ...attrs, key: String(key ?? i) }),
      )}
    </svg>
  );
}

/* Icons: Hugeicons Free (MIT, © Hugeicons, hugeicons.com) */
function ChevronIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M9 6s6 4.42 6 6-6 6-6 6" />
    </svg>
  );
}
function CheckIcon() {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path className="ui-dmc__tick" d="M5 14 8.5 17.5 19 6.5" pathLength={1} />
    </svg>
  );
}

/* ---------- Styles ---------- */

const css = /* css */ `
/* Colors and motion as variables – override them on .ui-dmc (see Theming in the docs) */
.ui-dmc {
  --ui-dmc-surface: var(--aui-overlay, var(--aui-surface, #ffffff));
  --ui-dmc-ink: var(--aui-ink, rgba(0, 0, 0, 0.88));
  --ui-dmc-text: var(--aui-muted, rgba(0, 0, 0, 0.61));
  --ui-dmc-icon: var(--aui-muted, rgba(0, 0, 0, 0.45));
  --ui-dmc-faint: var(--aui-placeholder, rgba(0, 0, 0, 0.27));
  --ui-dmc-line: var(--aui-border, rgba(0, 0, 0, 0.06));
  --ui-dmc-glow: rgba(0, 0, 0, 0.035);
  --ui-dmc-danger: var(--aui-red, #f04438);
  --ui-dmc-danger-glow: color-mix(in srgb, var(--aui-red, #f04438) 8%, transparent);
  --ui-dmc-shadow: var(--aui-shadow-popover, 0 0 0 0.5px var(--aui-border-strong, rgba(0, 0, 0, 0.1)), 0 12px 32px -8px rgba(0, 0, 0, 0.16));
  --ui-dmc-focus: var(--aui-focus, rgba(0, 96, 226, 0.7));
  --ui-dmc-ease: var(--aui-ease, cubic-bezier(0.22, 1, 0.36, 1));
  --ui-dmc-radius: calc(var(--aui-radius-lg, 10px) + 2px);
  --ui-dmc-row-radius: calc(var(--aui-radius-sm, 5px) + 3px);

  position: relative;
  display: inline-flex;
  min-width: 0;
}

/* The menu: a white card in the top layer, a 0.5px edge and a soft drop */
.ui-dmc__panel {
  position: fixed;
  inset: auto;
  z-index: 1000;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  min-width: 168px;
  /* Fits its content up to 320px – long items shorten with … (full text on hover) instead of stretching it; the width prop lifts the cap */
  max-width: min(320px, calc(100vw - 16px));
  max-height: calc(100vh - 16px);
  margin: 0;
  padding: 4px;
  border: 0;
  border-radius: var(--ui-dmc-radius);
  background: var(--ui-dmc-surface);
  box-shadow: var(--ui-dmc-shadow);
  color: var(--ui-dmc-ink);
  font-family: var(--aui-font, "Inter", "InterVariable", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif);
  font-size: 13px;
  line-height: 20px;
  font-weight: 400;
  text-align: start;
  overflow: visible;
  -webkit-font-smoothing: antialiased;
  transform-origin: top left;
  animation: ui-dmc-in 220ms var(--ui-dmc-ease) both;
}
.ui-dmc__panel::backdrop { display: none; }
.ui-dmc__panel[data-align="end"] { transform-origin: top right; }
.ui-dmc__panel[data-side="top"] { transform-origin: bottom left; animation-name: ui-dmc-in-up; }
.ui-dmc__panel[data-side="top"][data-align="end"] { transform-origin: bottom right; }
.ui-dmc__panel[data-side="right"] { transform-origin: top left; animation-name: ui-dmc-in-side; animation-duration: 160ms; }
.ui-dmc__panel[data-side="left"] { transform-origin: top right; animation-name: ui-dmc-in-side; animation-duration: 160ms; }

.ui-dmc__title { display: block; flex: none; padding: 6px 10px 4px; font-size: 12px; line-height: 16px; font-weight: 500; color: var(--ui-dmc-faint); }

/* The rows; a long menu scrolls inside the card */
.ui-dmc__list {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  align-content: start;
  gap: 1px;
  flex: 1 1 auto;
  min-height: 0;
  overflow-y: auto;
  overscroll-behavior: contain;
  scrollbar-width: thin;
}
.ui-dmc__list:focus { outline: none; }

/* One soft highlight that glides between the rows */
.ui-dmc__glow {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  border-radius: var(--ui-dmc-row-radius);
  background: var(--ui-dmc-glow);
  pointer-events: none;
  transition: transform 180ms var(--ui-dmc-ease), height 180ms var(--ui-dmc-ease), background-color 120ms ease;
}
.ui-dmc__glow[data-danger] { background: var(--ui-dmc-danger-glow); }

.ui-dmc__item {
  position: relative;
  display: flex;
  align-items: center;
  gap: 9px;
  box-sizing: border-box;
  min-width: 0;
  min-height: 32px;
  padding: 0 10px;
  border-radius: var(--ui-dmc-row-radius);
  color: var(--ui-dmc-text);
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
  transition: color 120ms ease;
}
.ui-dmc__item[data-rich] { min-height: 44px; padding: 6px 10px; }
.ui-dmc__item:focus { outline: none; color: var(--ui-dmc-ink); }
.ui-dmc__item:focus-visible { outline: 2px solid var(--ui-dmc-focus); outline-offset: -2px; }
/* The item whose submenu is open keeps a highlight while the focus is in the submenu */
.ui-dmc__item[data-open] { color: var(--ui-dmc-ink); }
.ui-dmc__item[data-open]:not(:focus) { background: var(--ui-dmc-glow); }
.ui-dmc__item[aria-checked="true"] { color: var(--ui-dmc-ink); font-weight: 500; }
.ui-dmc__item[data-danger], .ui-dmc__item[data-danger]:focus { color: var(--ui-dmc-danger); }
.ui-dmc__item[aria-disabled="true"] { color: var(--ui-dmc-faint); cursor: default; }

.ui-dmc__icon { display: grid; flex: none; place-items: center; width: 16px; height: 16px; color: var(--ui-dmc-icon); transition: color 120ms ease; }
.ui-dmc__icon > svg { display: block; width: 15px; height: 15px; }
.ui-dmc__icon[data-logo] > svg, .ui-dmc__icon[data-logo] > img { width: 14px; height: 14px; }
.ui-dmc__item:focus .ui-dmc__icon:not([data-logo]), .ui-dmc__item[data-open] .ui-dmc__icon:not([data-logo]) { color: var(--ui-dmc-ink); }
.ui-dmc__item[data-danger] .ui-dmc__icon { color: var(--ui-dmc-danger); }
.ui-dmc__item[aria-disabled="true"] .ui-dmc__icon { opacity: 0.5; color: var(--ui-dmc-icon); }

/* Long names stay on one line – the full name on hover */
.ui-dmc__label { display: grid; flex: 1; gap: 1px; min-width: 0; }
.ui-dmc__text, .ui-dmc__desc { overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.ui-dmc__desc { font-size: 12px; line-height: 16px; font-weight: 400; color: var(--ui-dmc-faint); }

.ui-dmc__kbd { flex: none; padding-left: 16px; font: inherit; font-size: 12px; font-weight: 400; letter-spacing: 0.04em; color: var(--ui-dmc-faint); }
.ui-dmc__more, .ui-dmc__check { display: grid; flex: none; place-items: center; width: 14px; height: 14px; }
.ui-dmc__more { color: var(--ui-dmc-faint); }
.ui-dmc__check { color: var(--ui-dmc-ink); }
.ui-dmc__more > svg, .ui-dmc__check > svg { display: block; width: 14px; height: 14px; }
.ui-dmc__tick { stroke-dasharray: 1; animation: ui-dmc-draw 280ms 40ms var(--ui-dmc-ease) both; }

.ui-dmc__sep { display: block; height: 1px; margin: 4px 6px; background: var(--ui-dmc-line); }
.ui-dmc__empty { display: block; padding: 6px 8px; color: var(--ui-dmc-faint); }

@keyframes ui-dmc-in { from { opacity: 0; transform: scale(0.96) translateY(-4px); } }
@keyframes ui-dmc-in-up { from { opacity: 0; transform: scale(0.96) translateY(4px); } }
@keyframes ui-dmc-in-side { from { opacity: 0; transform: scale(0.98); } }
@keyframes ui-dmc-draw { from { stroke-dashoffset: 1; } to { stroke-dashoffset: 0; } }

/* Touch screens: rows a finger can hit */
@media (pointer: coarse) {
  .ui-dmc__item { min-height: 40px; }
  .ui-dmc__item[data-rich] { min-height: 48px; }
}

@media (prefers-reduced-motion: reduce) {
  .ui-dmc__panel, .ui-dmc__tick { animation-duration: 1ms; }
  .ui-dmc__glow, .ui-dmc__item, .ui-dmc__icon { transition-duration: 1ms; }
}

/* Dark mode: follows a .dark class or data-theme="dark" on any ancestor, e.g. <html> */
:where(.dark, [data-theme="dark"]) .ui-dmc {
  --ui-dmc-surface: var(--aui-overlay, var(--aui-surface, #1b1b1e));
  --ui-dmc-ink: var(--aui-ink, rgba(255, 255, 255, 0.9));
  --ui-dmc-text: var(--aui-muted, rgba(255, 255, 255, 0.62));
  --ui-dmc-icon: var(--aui-muted, rgba(255, 255, 255, 0.45));
  --ui-dmc-faint: var(--aui-placeholder, rgba(255, 255, 255, 0.28));
  --ui-dmc-line: var(--aui-border, rgba(255, 255, 255, 0.08));
  --ui-dmc-glow: rgba(255, 255, 255, 0.06);
  --ui-dmc-danger: var(--aui-red, #ff6b5f);
  --ui-dmc-danger-glow: color-mix(in srgb, var(--aui-red, #ff6b5f) 14%, transparent);
  --ui-dmc-shadow: var(--aui-shadow-popover, 0 0 0 0.5px var(--aui-border-strong, rgba(255, 255, 255, 0.14)), 0 12px 32px -8px rgba(0, 0, 0, 0.6));
  --ui-dmc-focus: var(--aui-focus, rgba(77, 141, 255, 0.8));
  color-scheme: dark;
}
`;
