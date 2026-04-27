"use client";

import { useMemo, useState } from "react";
import { CheckIcon } from "@phosphor-icons/react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

import {
  EASE_OUT,
  NOTIFICATION_GROUPS,
  type NotificationFlags,
  type NotificationGroup,
  type NotificationItem,
  type NotificationKey,
} from "../../helpers";

interface Props {
  values: NotificationFlags;
  onToggle: (key: NotificationKey, value: boolean) => void;
  onSetAllInGroup: (keys: NotificationKey[], value: boolean) => void;
  index?: number;
}

type TabId = "all" | NotificationGroup["id"];

export function NotificationsCard({
  values,
  onToggle,
  onSetAllInGroup,
  index = 0,
}: Props) {
  const reduceMotion = useReducedMotion();
  const [activeTab, setActiveTab] = useState<TabId>("all");

  const tabs = useMemo<{ id: TabId; label: string; count: number }[]>(() => {
    const enabledByGroup = NOTIFICATION_GROUPS.map((group) => {
      const enabled = group.items.filter((item) => values[item.key]).length;
      return { id: group.id, label: group.title, count: enabled };
    });
    const allEnabled = enabledByGroup.reduce((sum, g) => sum + g.count, 0);
    return [
      { id: "all", label: "All", count: allEnabled },
      ...enabledByGroup,
    ];
  }, [values]);

  const visibleGroups = useMemo<NotificationGroup[]>(() => {
    if (activeTab === "all") return NOTIFICATION_GROUPS;
    return NOTIFICATION_GROUPS.filter((g) => g.id === activeTab);
  }, [activeTab]);

  return (
    <motion.section
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: EASE_OUT, delay: 0.04 + index * 0.05 }}
      className="overflow-hidden rounded-[18px] border border-zinc-200 bg-white shadow-[0_1px_2px_rgba(15,15,20,0.04)] transition-shadow duration-200 ease-out focus-within:shadow-[0_8px_28px_-12px_rgba(15,15,20,0.12)]"
    >
      <div className="flex flex-col gap-4 px-6 pt-6">
        <div className="flex flex-col gap-1">
          <Text
            variant="small-medium"
            as="span"
            className="uppercase tracking-[0.08em] text-zinc-400"
          >
            Notifications
          </Text>
          <Text variant="h4" as="h2" className="text-[#1F1F20]">
            What should we tell you about?
          </Text>
          <Text variant="small" className="text-zinc-500">
            Filter by category, or scan everything in one place.
          </Text>
        </div>

        <CategoryTabs
          tabs={tabs}
          activeTab={activeTab}
          onTabChange={setActiveTab}
        />
      </div>

      <div className="px-2 pb-2 pt-2 sm:px-3">
        <AnimatePresence initial={false} mode="wait">
          <motion.div
            key={activeTab}
            initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduceMotion ? { opacity: 0 } : { opacity: 0, y: -6 }}
            transition={{ duration: 0.2, ease: EASE_OUT }}
            className="flex flex-col gap-1"
          >
            {visibleGroups.map((group) => (
              <NotificationGroupSection
                key={group.id}
                group={group}
                values={values}
                onToggle={onToggle}
                onSetAllInGroup={onSetAllInGroup}
              />
            ))}
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.section>
  );
}

function CategoryTabs({
  tabs,
  activeTab,
  onTabChange,
}: {
  tabs: { id: TabId; label: string; count: number }[];
  activeTab: TabId;
  onTabChange: (id: TabId) => void;
}) {
  return (
    <div
      role="tablist"
      aria-label="Notification categories"
      className="-mx-1 flex gap-1 overflow-x-auto pb-1"
    >
      {tabs.map((tab) => {
        const isActive = activeTab === tab.id;
        return (
          <button
            key={tab.id}
            role="tab"
            type="button"
            aria-selected={isActive}
            onClick={() => onTabChange(tab.id)}
            className={cn(
              "relative flex shrink-0 items-center gap-1.5 rounded-full px-3 py-1.5 text-zinc-600 transition-colors duration-150 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2",
              isActive ? "text-zinc-900" : "hover:text-zinc-900",
            )}
          >
            {isActive ? (
              <motion.span
                layoutId="notif-tab-pill"
                aria-hidden
                className="absolute inset-0 rounded-full bg-zinc-900"
                transition={{ type: "spring", stiffness: 480, damping: 36 }}
              />
            ) : null}
            <span
              className={cn(
                "relative z-10 whitespace-nowrap text-[13px] font-medium",
                isActive && "text-white",
              )}
            >
              {tab.label}
            </span>
            <span
              className={cn(
                "relative z-10 inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-full px-1 text-[11px] font-medium tabular-nums",
                isActive ? "bg-white/20 text-white" : "bg-zinc-100 text-zinc-500",
              )}
            >
              {tab.count}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function NotificationGroupSection({
  group,
  values,
  onToggle,
  onSetAllInGroup,
}: {
  group: NotificationGroup;
  values: NotificationFlags;
  onToggle: (key: NotificationKey, value: boolean) => void;
  onSetAllInGroup: (keys: NotificationKey[], value: boolean) => void;
}) {
  const Icon = group.icon;
  const itemKeys = group.items.map((i) => i.key);
  const allOn = itemKeys.every((k) => values[k]);
  const anyOn = itemKeys.some((k) => values[k]);

  return (
    <div className="rounded-[14px]">
      <div className="flex items-center gap-3 px-4 pb-2 pt-4">
        <span
          className={cn(
            "inline-flex h-9 w-9 items-center justify-center rounded-full bg-gradient-to-br",
            group.accent,
          )}
        >
          <Icon size={18} weight="duotone" />
        </span>
        <div className="flex min-w-0 flex-1 flex-col">
          <Text variant="large-medium" as="h3" className="text-[#1F1F20]">
            {group.title}
          </Text>
          <Text variant="small" className="text-zinc-500">
            {group.caption}
          </Text>
        </div>
        <button
          type="button"
          onClick={() => onSetAllInGroup(itemKeys, !allOn)}
          aria-pressed={allOn}
          className={cn(
            "flex shrink-0 items-center gap-1 rounded-full border px-2.5 py-1 text-[12px] font-medium transition-colors duration-150 ease-out focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-violet-400 focus-visible:ring-offset-2",
            allOn
              ? "border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100"
              : anyOn
                ? "border-zinc-200 bg-white text-zinc-700 hover:bg-zinc-50"
                : "border-zinc-200 bg-white text-zinc-500 hover:bg-zinc-50",
          )}
        >
          {allOn ? (
            <>
              <CheckIcon size={12} weight="bold" /> All on
            </>
          ) : (
            "Enable all"
          )}
        </button>
      </div>

      <div className="flex flex-col">
        {group.items.map((item, i) => (
          <NotificationRow
            key={item.key}
            item={item}
            value={values[item.key]}
            onChange={(checked) => onToggle(item.key, checked)}
            isLast={i === group.items.length - 1}
          />
        ))}
      </div>
    </div>
  );
}

function NotificationRow({
  item,
  value,
  onChange,
  isLast,
}: {
  item: NotificationItem;
  value: boolean;
  onChange: (value: boolean) => void;
  isLast: boolean;
}) {
  const id = `notif-${item.key}`;
  return (
    <label
      htmlFor={id}
      className={cn(
        "group flex cursor-pointer items-center justify-between gap-4 rounded-[12px] px-4 py-3 transition-colors duration-150 ease-out hover:bg-zinc-50",
        !isLast && "mb-0.5",
      )}
    >
      <div className="flex min-w-0 flex-col">
        <Text variant="body-medium" as="span" className="text-[#1F1F20]">
          {item.title}
        </Text>
        <Text variant="small" as="span" className="text-zinc-500">
          {item.description}
        </Text>
      </div>
      <Switch
        id={id}
        aria-label={item.title}
        checked={value}
        onCheckedChange={onChange}
      />
    </label>
  );
}
