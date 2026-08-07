"use client";

import { useState } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import {
  TabsLine,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
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
  index?: number;
}

type TabId = NotificationGroup["id"];

export function NotificationsCard({ values, onToggle, index = 0 }: Props) {
  const reduceMotion = useReducedMotion();
  const [activeTab, setActiveTab] = useState<TabId>(NOTIFICATION_GROUPS[0].id);

  const tabs: { id: TabId; label: string; count: number }[] =
    NOTIFICATION_GROUPS.map((group) => ({
      id: group.id,
      label: group.title,
      count: group.items.filter((item) => values[item.key]).length,
    }));

  const visibleGroups: NotificationGroup[] = NOTIFICATION_GROUPS.filter(
    (g) => g.id === activeTab,
  );

  return (
    <motion.section
      initial={reduceMotion ? false : { opacity: 0, y: 12 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : {
              duration: 0.32,
              ease: EASE_OUT,
              delay: 0.04 + index * 0.05,
            }
      }
      className="flex w-full flex-col gap-2 pt-0"
    >
      <div className="flex flex-col gap-1 px-4">
        <Text variant="body-medium" as="span" className="text-textBlack">
          Notifications
        </Text>
        <Text variant="small" className="text-zinc-500">
          Filter by category, or scan everything in one place.
        </Text>
      </div>

      <div className="flex flex-col gap-3 rounded-[18px] border border-zinc-200 bg-white px-4 pb-4 pt-0 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
        <TabsLine
          value={activeTab}
          onValueChange={(v) => setActiveTab(v as TabId)}
        >
          <TabsLineList>
            {tabs.map((tab) => (
              <TabsLineTrigger key={tab.id} value={tab.id}>
                {tab.label}
                <span className="ml-2 inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-full bg-zinc-100 px-1 text-[11px] font-medium tabular-nums text-zinc-500">
                  {tab.count}
                </span>
              </TabsLineTrigger>
            ))}
          </TabsLineList>
        </TabsLine>

        <AnimatePresence initial={false} mode="wait">
          <motion.div
            key={activeTab}
            initial={reduceMotion ? false : { opacity: 0, y: 6 }}
            animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
            exit={reduceMotion ? undefined : { opacity: 0, y: -6 }}
            transition={
              reduceMotion ? undefined : { duration: 0.2, ease: EASE_OUT }
            }
            className="flex flex-col gap-1"
          >
            {visibleGroups.map((group) => (
              <NotificationGroupSection
                key={group.id}
                group={group}
                values={values}
                onToggle={onToggle}
              />
            ))}
          </motion.div>
        </AnimatePresence>
      </div>
    </motion.section>
  );
}

function NotificationGroupSection({
  group,
  values,
  onToggle,
}: {
  group: NotificationGroup;
  values: NotificationFlags;
  onToggle: (key: NotificationKey, value: boolean) => void;
}) {
  return (
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
        <Text variant="body-medium" as="span" className="text-textBlack">
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
