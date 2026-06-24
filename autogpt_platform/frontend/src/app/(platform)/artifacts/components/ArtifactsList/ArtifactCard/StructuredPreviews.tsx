"use client";

import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import { EnvelopeSimpleIcon, PhoneIcon } from "@phosphor-icons/react";
import { getFilePreviewUrl } from "../helpers";

// Whole file (kind is gated < 110KB) via the preview endpoint, which sets
// browser cache headers — unlike /download (no-store).
const STRUCTURED_PREVIEW_BYTES = 110_000;
import {
  parseIcs,
  parseVcard,
  type IcsPreview as IcsData,
  type VcardPreview as VcardData,
} from "../parsers";
import { Fallback, LoadingPlaceholder, useFileText } from "./PreviewParts";

interface PreviewProps {
  file: WorkspaceFileItem;
  onError: () => void;
}

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

interface DateParts {
  year: number;
  month: number;
  day: number;
  hour: number | null;
  minute: number | null;
}

function parseDateTime(value: string | undefined): DateParts | null {
  const digits = (value ?? "").replace(/[^0-9]/g, "");
  if (digits.length < 8) return null;
  const month = Number(digits.slice(4, 6));
  const day = Number(digits.slice(6, 8));
  if (month < 1 || month > 12 || day < 1 || day > 31) return null;
  const hasTime = digits.length >= 12;
  return {
    year: Number(digits.slice(0, 4)),
    month,
    day,
    hour: hasTime ? Number(digits.slice(8, 10)) : null,
    minute: hasTime ? Number(digits.slice(10, 12)) : null,
  };
}

function weekdayLabel(dt: DateParts): string {
  return WEEKDAYS[
    new Date(Date.UTC(dt.year, dt.month - 1, dt.day)).getUTCDay()
  ];
}

function clock(hour: number, minute: number): string {
  return `${hour % 12 === 0 ? 12 : hour % 12}:${String(minute).padStart(2, "0")}`;
}

function meridiem(hour: number): string {
  return hour >= 12 ? "PM" : "AM";
}

function formatTimeRange(
  start: string | undefined,
  end: string | undefined,
): string | null {
  const s = parseDateTime(start);
  if (!s || s.hour === null || s.minute === null) return null;
  const e = parseDateTime(end);
  if (e && e.hour !== null && e.minute !== null) {
    // Collapse the meridiem when both ends share it: "3:30–4:30 PM".
    if (meridiem(s.hour) === meridiem(e.hour)) {
      return `${clock(s.hour, s.minute)}–${clock(e.hour, e.minute)} ${meridiem(e.hour)}`;
    }
    return `${clock(s.hour, s.minute)} ${meridiem(s.hour)} – ${clock(e.hour, e.minute)} ${meridiem(e.hour)}`;
  }
  return `${clock(s.hour, s.minute)} ${meridiem(s.hour)}`;
}

export function IcsPreview({ file, onError }: PreviewProps) {
  const text = useFileText(
    getFilePreviewUrl(file.id, { bytes: STRUCTURED_PREVIEW_BYTES }),
    onError,
  );
  if (text === null) return <LoadingPlaceholder file={file} />;

  const event = parseIcs(text);
  if (!event) return <Fallback file={file} />;
  return <EventCard event={event} />;
}

function EventCard({ event }: { event: IcsData }) {
  const date = parseDateTime(event.start);
  const timeRange = formatTimeRange(event.start, event.end);
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-0.5 bg-white p-3 text-center">
      {date ? (
        <>
          <span className="text-sm font-bold uppercase leading-none tracking-wide text-red-500">
            {weekdayLabel(date)}
          </span>
          <span className="text-4xl font-bold leading-none text-zinc-900">
            {date.day}
          </span>
        </>
      ) : null}
      <span className="mt-1.5 line-clamp-2 text-sm font-medium text-zinc-800">
        {event.summary ?? "Event"}
      </span>
      {timeRange ? (
        <span className="text-xs text-zinc-600">{timeRange}</span>
      ) : null}
    </div>
  );
}

export function VcardPreview({ file, onError }: PreviewProps) {
  const text = useFileText(
    getFilePreviewUrl(file.id, { bytes: STRUCTURED_PREVIEW_BYTES }),
    onError,
  );
  if (text === null) return <LoadingPlaceholder file={file} />;

  const contact = parseVcard(text);
  if (!contact) return <Fallback file={file} />;
  return <ContactCard contact={contact} />;
}

function initials(name: string): string {
  return name
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("");
}

function ContactCard({ contact }: { contact: VcardData }) {
  return (
    <div className="flex h-full w-full flex-col items-center justify-center gap-1.5 bg-white p-3 text-center">
      {contact.photo ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={contact.photo}
          alt={contact.name ?? "Contact"}
          className="h-14 w-14 rounded-full object-cover"
        />
      ) : (
        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-zinc-200 text-lg font-semibold text-zinc-600">
          {initials(contact.name ?? "")}
        </div>
      )}
      <span className="truncate text-sm font-medium text-zinc-900">
        {contact.name}
      </span>
      {contact.title || contact.org ? (
        <span className="truncate text-xs text-zinc-500">
          {[contact.title, contact.org].filter(Boolean).join(" · ")}
        </span>
      ) : null}
      {contact.tel ? (
        <span className="flex items-center gap-1 text-xs text-zinc-500">
          <PhoneIcon size={12} className="shrink-0" />
          <span className="truncate">{contact.tel}</span>
        </span>
      ) : null}
      {contact.email ? (
        <span className="flex items-center gap-1 text-xs text-zinc-500">
          <EnvelopeSimpleIcon size={12} className="shrink-0" />
          <span className="truncate">{contact.email}</span>
        </span>
      ) : null}
    </div>
  );
}
