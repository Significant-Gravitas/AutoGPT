"use client";

import { useId } from "react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { LinkSimpleIcon, PlusIcon, TrashIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";

import { MAX_LINKS } from "../../helpers";

interface Props {
  links: string[];
  onChange: (index: number, value: string) => void;
  onAdd: () => void;
  onRemove: (index: number) => void;
}

const EASE_OUT = [0.16, 1, 0.3, 1] as const;

export function LinksSection({ links, onChange, onAdd, onRemove }: Props) {
  const reduceMotion = useReducedMotion();
  const baseId = useId();
  const canAdd = links.length < MAX_LINKS;

  return (
    <motion.div
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: EASE_OUT, delay: 0.12 }}
      className="flex flex-col gap-5 rounded-[16px] border border-zinc-200 bg-white p-6 shadow-[0_1px_2px_rgba(15,15,20,0.04)]"
    >
      <div className="flex flex-col gap-1">
        <Text variant="h4" as="h2" className="text-[#1F1F20]">
          Your links
        </Text>
        <Text variant="small" className="text-zinc-500">
          Add up to {MAX_LINKS} links. Site, GitHub, X, anything you want
          visible.
        </Text>
      </div>

      <div className="flex flex-col gap-3">
        <AnimatePresence initial={false}>
          {links.map((link, index) => (
            <motion.div
              key={`${baseId}-link-${index}`}
              layout={!reduceMotion}
              initial={
                reduceMotion
                  ? { opacity: 0 }
                  : { opacity: 0, height: 0, y: -6 }
              }
              animate={
                reduceMotion
                  ? { opacity: 1 }
                  : { opacity: 1, height: "auto", y: 0 }
              }
              exit={
                reduceMotion
                  ? { opacity: 0 }
                  : { opacity: 0, height: 0, y: -6 }
              }
              transition={{ duration: 0.22, ease: EASE_OUT }}
              className="overflow-visible"
            >
              <div className="flex items-center gap-2">
                <div className="flex flex-1 items-center gap-2 rounded-3xl border border-zinc-200 bg-white px-4 transition-colors duration-150 focus-within:border-purple-400 focus-within:ring-1 focus-within:ring-purple-400">
                  <LinkSimpleIcon
                    size={16}
                    weight="regular"
                    className="text-zinc-400"
                  />
                  <input
                    type="url"
                    value={link}
                    placeholder="https://"
                    aria-label={`Link ${index + 1}`}
                    onChange={(e) => onChange(index, e.target.value)}
                    className="h-[2.875rem] w-full border-none bg-transparent text-sm text-black placeholder:text-zinc-400 focus:outline-none"
                  />
                </div>
                <Button
                  variant="icon"
                  size="icon"
                  aria-label={`Remove link ${index + 1}`}
                  onClick={() => onRemove(index)}
                >
                  <TrashIcon size={16} />
                </Button>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>

        {links.length === 0 ? (
          <Text variant="small" className="text-zinc-400">
            No links yet.
          </Text>
        ) : null}

        <div>
          <Button
            variant="ghost"
            size="small"
            leftIcon={<PlusIcon size={16} />}
            onClick={onAdd}
            disabled={!canAdd}
          >
            {canAdd ? "Add link" : `Limit of ${MAX_LINKS} reached`}
          </Button>
        </div>
      </div>
    </motion.div>
  );
}
