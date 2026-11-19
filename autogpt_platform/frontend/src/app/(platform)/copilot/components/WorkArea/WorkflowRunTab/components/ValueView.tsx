"use client";

import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";
import {
  countLabel,
  dateOf,
  headOf,
  hostOf,
  isEmpty,
  isLink,
  isRecord,
  labelOf,
  plainText,
} from "../value";

const LONG_TEXT = 220;
const FIRST_ITEMS = 3;
const MAX_DEPTH = 3;

const linkClass =
  "rounded-sm underline underline-offset-2 outline-none focus-visible:ring-2 focus-visible:ring-focus-inbox";
const moreClass =
  "self-start rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-focus-inbox";

interface Props {
  value: unknown;
  depth?: number;
}

/** A block's value drawn the way people read it: text as text, a link as a
 *  link, a list as its items and a record as named fields. */
export function ValueView({ value, depth = 0 }: Props) {
  if (isEmpty(value)) {
    return (
      <Text variant="small" as="span" tone="faint">
        Empty
      </Text>
    );
  }
  if (typeof value === "boolean") {
    return (
      <Text variant="small" as="span" tone="ink">
        {value ? "Yes" : "No"}
      </Text>
    );
  }
  if (typeof value === "number") {
    return (
      <Text variant="small" as="span" tone="ink" tabular>
        {value.toLocaleString()}
      </Text>
    );
  }
  if (typeof value === "string") return <TextValue value={value} />;
  if (depth >= MAX_DEPTH) return <TextValue value={JSON.stringify(value)} />;
  if (Array.isArray(value)) return <ListValue items={value} depth={depth} />;
  if (isRecord(value)) return <Fields value={value} depth={depth} />;
  return <TextValue value={String(value)} />;
}

function LinkValue({ link }: { link: string }) {
  return (
    <Text variant="small" as="span" tone="ink" className="min-w-0 truncate">
      <a
        href={link}
        target="_blank"
        rel="noreferrer"
        title={link}
        className={linkClass}
      >
        {hostOf(link)}
      </a>
    </Text>
  );
}

function TextValue({ value }: { value: string }) {
  const [isOpen, setIsOpen] = useState(false);
  if (isLink(value)) return <LinkValue link={value.trim()} />;
  const date = dateOf(value);
  const body = date ?? plainText(value);
  const isLong = body.length > LONG_TEXT;
  return (
    <span className="flex min-w-0 flex-col gap-0.5">
      <Text
        variant="small"
        as="span"
        tone="ink"
        className={`break-words whitespace-pre-wrap ${isLong && !isOpen ? "line-clamp-3" : ""}`}
      >
        {body}
      </Text>
      {isLong ? (
        <button
          type="button"
          onClick={() => setIsOpen(!isOpen)}
          aria-expanded={isOpen}
          className={moreClass}
        >
          <Text variant="small" as="span" tone="quiet">
            {isOpen ? "Show less" : "Show more"}
          </Text>
        </button>
      ) : null}
    </span>
  );
}

interface FieldsProps {
  value: Record<string, unknown>;
  depth: number;
  /** Fields already shown elsewhere, such as an item's title. */
  skip?: (string | null)[];
}

function Fields({ value, depth, skip = [] }: FieldsProps) {
  const entries = Object.entries(value).filter(
    ([key, item]) => !skip.includes(key) && !isEmpty(item),
  );
  if (entries.length === 0) return null;
  return (
    <dl className="flex min-w-0 flex-col gap-1.5">
      {entries.map(([key, item]) => (
        <div key={key} className="flex min-w-0 flex-col gap-0.5">
          <Text variant="small" as="dt" tone="quiet" className="truncate">
            {labelOf(key)}
          </Text>
          <dd className="min-w-0">
            <ValueView value={item} depth={depth + 1} />
          </dd>
        </div>
      ))}
    </dl>
  );
}

interface ListProps {
  items: unknown[];
  depth: number;
}

function ListValue({ items, depth }: ListProps) {
  const [showAll, setShowAll] = useState(false);
  const shown = showAll ? items : items.slice(0, FIRST_ITEMS);
  const isPlain = items.every((item) => !item || typeof item !== "object");
  return (
    <div className="flex min-w-0 flex-col gap-1.5">
      <Text variant="small" as="span" tone="quiet">
        {countLabel(items.length, "item", "items")}
      </Text>
      <ul className="flex min-w-0 flex-col gap-1.5">
        {shown.map((item, index) => (
          <li
            // eslint-disable-next-line react/no-array-index-key -- a run's values never reorder
            key={index}
            className={
              isPlain ? "min-w-0" : "min-w-0 rounded-lg bg-surface px-2.5 py-2"
            }
          >
            {isRecord(item) ? (
              <ItemCard item={item} depth={depth} />
            ) : (
              <ValueView value={item} depth={depth + 1} />
            )}
          </li>
        ))}
      </ul>
      {items.length > FIRST_ITEMS ? (
        <button
          type="button"
          onClick={() => setShowAll(!showAll)}
          aria-expanded={showAll}
          className={moreClass}
        >
          <Text variant="small" as="span" tone="quiet">
            {showAll
              ? "Show fewer"
              : `Show all ${items.length.toLocaleString()}`}
          </Text>
        </button>
      ) : null}
    </div>
  );
}

interface ItemProps {
  item: Record<string, unknown>;
  depth: number;
}

/** One record of a list: its name and where it points, with the rest of
 *  its fields a click away. */
function ItemCard({ item, depth }: ItemProps) {
  const [isOpen, setIsOpen] = useState(false);
  const head = headOf(item);
  if (!head.title) return <Fields value={item} depth={depth + 1} />;
  const rest = Object.keys(item).filter(
    (key) =>
      key !== head.titleKey && key !== head.linkKey && !isEmpty(item[key]),
  );
  return (
    <div className="flex min-w-0 flex-col gap-1">
      <Text
        variant="small-medium"
        as="span"
        tone="ink"
        className="text-pretty break-words"
      >
        {head.title}
      </Text>
      <span className="flex min-w-0 items-center gap-2">
        {head.link ? <LinkValue link={head.link} /> : null}
        {rest.length > 0 ? (
          <button
            type="button"
            onClick={() => setIsOpen(!isOpen)}
            aria-expanded={isOpen}
            className={`${moreClass} shrink-0 self-auto`}
          >
            <Text variant="small" as="span" tone="quiet">
              {isOpen
                ? "Hide details"
                : countLabel(rest.length, "more field", "more fields")}
            </Text>
          </button>
        ) : null}
      </span>
      {isOpen ? (
        <div className="pt-1">
          <Fields
            value={item}
            depth={depth + 1}
            skip={[head.titleKey, head.linkKey]}
          />
        </div>
      ) : null}
    </div>
  );
}
