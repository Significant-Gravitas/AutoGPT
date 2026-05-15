"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { LinkSimpleIcon, PlusIcon, TrashIcon } from "@phosphor-icons/react";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import { type LinkRow, MAX_LINKS } from "../../helpers";

interface Props {
  links: LinkRow[];
  onChange: (index: number, value: string) => void;
  onAdd: () => void;
  onRemove: (index: number) => void;
}

const EASE_OUT = [0.16, 1, 0.3, 1] as const;

export function LinksSection({ links, onChange, onAdd, onRemove }: Props) {
  const reduceMotion = useReducedMotion();
  const canAdd = links.length < MAX_LINKS;

  return (
    <motion.div
      initial={reduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.32, ease: EASE_OUT, delay: 0.12 }}
      className="flex flex-col gap-4"
    >
      <div className="flex flex-col gap-1 px-4">
        <Text variant="body-medium" as="span" className="text-black">
          Your links
        </Text>
        <Text variant="small" className="text-zinc-500">
          Add up to {MAX_LINKS} links. Site, GitHub, X, anything you want
          visible.
        </Text>
      </div>

      <div className="flex flex-col gap-3">
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
          <AnimatePresence initial={false}>
            {links.map((link, index) => (
              <motion.div
                key={link.id}
                layout={!reduceMotion}
                initial={
                  reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.97 }
                }
                animate={{ opacity: 1, scale: 1 }}
                exit={
                  reduceMotion ? { opacity: 0 } : { opacity: 0, scale: 0.97 }
                }
                transition={{ duration: 0.18, ease: EASE_OUT }}
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
                      value={link.value}
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
        </div>

        {links.length === 0 ? (
          <Text variant="small" className="px-4 text-zinc-400">
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
