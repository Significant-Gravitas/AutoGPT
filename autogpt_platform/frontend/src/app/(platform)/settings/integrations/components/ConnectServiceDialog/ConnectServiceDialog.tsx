"use client";

import { useLayoutEffect, useRef, useState } from "react";
import {
  AnimatePresence,
  MotionConfig,
  motion,
  useReducedMotion,
} from "framer-motion";
import {
  MagnifyingGlassIcon,
  PlugIcon,
  PlusIcon,
} from "@phosphor-icons/react";
import Image from "next/image";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { Text } from "@/components/atoms/Text/Text";
import { ScrollArea } from "@/components/ui/scroll-area";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";

import { DetailView } from "./components/DetailView/DetailView";
import { ConnectableProvider } from "./helpers";
import { useConnectServiceDialog } from "./useConnectServiceDialog";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const TRANSITION = { duration: 0.22, ease: [0, 0, 0.2, 1] as const };
const HEIGHT_TRANSITION = {
  duration: 0.28,
  ease: [0.32, 0.72, 0, 1] as const,
};

const stepVariants = {
  initial: (direction: number) => ({ x: 24 * direction, opacity: 0 }),
  active: { x: 0, opacity: 1 },
  exit: (direction: number) => ({ x: -24 * direction, opacity: 0 }),
};

const reducedVariants = {
  initial: { opacity: 0 },
  active: { opacity: 1 },
  exit: { opacity: 0 },
};

function useMeasuredHeight<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [height, setHeight] = useState<number | undefined>(undefined);

  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;
    setHeight(node.offsetHeight);
    const observer = new ResizeObserver((entries) => {
      const next = entries[0]?.contentRect.height;
      if (typeof next === "number") setHeight(next);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return [ref, height] as const;
}

export function ConnectServiceDialog({ open, onOpenChange }: Props) {
  const {
    query,
    setQuery,
    providers,
    isLoading,
    isError,
    error,
    refetch,
    view,
    direction,
    selectedProvider,
    handleSelect,
    handleBack,
    handleSuccess,
  } = useConnectServiceDialog({ open, onOpenChange });

  const reduceMotion = useReducedMotion();
  const variants = reduceMotion ? reducedVariants : stepVariants;
  const [contentRef, contentHeight] = useMeasuredHeight<HTMLDivElement>();

  return (
    <Dialog
      title="Connect a service"
      styling={{ maxWidth: "40rem" }}
      controlled={{
        isOpen: open,
        set: (next) => onOpenChange(next),
      }}
    >
      <Dialog.Content>
        <MotionConfig transition={TRANSITION}>
          <motion.div
            className="relative overflow-hidden"
            animate={{ height: contentHeight ?? "auto" }}
            transition={reduceMotion ? { duration: 0 } : HEIGHT_TRANSITION}
            style={{ willChange: "height" }}
          >
            <div ref={contentRef}>
              <AnimatePresence
                mode="wait"
                initial={false}
                custom={direction}
              >
                {view === "list" ? (
                  <motion.div
                    key="list"
                    custom={direction}
                    variants={variants}
                    initial="initial"
                    animate="active"
                    exit="exit"
                  >
                    {isLoading ? (
                      <ListLoading />
                    ) : isError ? (
                      <ErrorCard
                        context="providers"
                        responseError={
                          error instanceof Error
                            ? { message: error.message }
                            : undefined
                        }
                        onRetry={() => refetch()}
                      />
                    ) : (
                      <ListView
                        query={query}
                        setQuery={setQuery}
                        providers={providers}
                        onSelect={handleSelect}
                      />
                    )}
                  </motion.div>
                ) : (
                  <motion.div
                    key={`detail-${selectedProvider?.id}`}
                    custom={direction}
                    variants={variants}
                    initial="initial"
                    animate="active"
                    exit="exit"
                  >
                    <DetailView
                      provider={selectedProvider!}
                      onBack={handleBack}
                      onSuccess={handleSuccess}
                    />
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        </MotionConfig>
      </Dialog.Content>
    </Dialog>
  );
}

interface ListViewProps {
  query: string;
  setQuery: (next: string) => void;
  providers: ConnectableProvider[];
  onSelect: (id: string) => void;
}

function ListView({ query, setQuery, providers, onSelect }: ListViewProps) {
  return (
    <div className="flex flex-col gap-4">
      <Text variant="body" className="text-[#505057]">
        Pick a service to connect an API key or authorize with OAuth.
      </Text>

      <div className="relative w-full">
        <MagnifyingGlassIcon
          size={20}
          className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-[#83838C]"
        />
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search services..."
          aria-label="Search services"
          className="h-[46px] w-full rounded-3xl border border-[#DADADC] bg-white pl-12 pr-4 text-sm leading-[22px] text-[#1F1F20] placeholder:text-[#83838C] focus:border-purple-400 focus:outline-none focus:ring-1 focus:ring-purple-400"
        />
      </div>

      {providers.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-[#DADADC] py-10 text-center">
          <PlugIcon size={24} className="text-[#83838C]" />
          <Text variant="body" className="text-[#505057]">
            {query.trim()
              ? `No services match "${query.trim()}"`
              : "No services available"}
          </Text>
        </div>
      ) : (
        <div className="relative">
          <ScrollArea className="h-[380px] pr-2">
            <ul className="flex flex-col gap-2 pb-4">
              {providers.map((provider) => (
                <li key={provider.id}>
                  <ProviderRow provider={provider} onSelect={onSelect} />
                </li>
              ))}
            </ul>
          </ScrollArea>
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-white to-transparent"
          />
        </div>
      )}
    </div>
  );
}

function ListLoading() {
  return (
    <div className="flex flex-col gap-4">
      <Skeleton className="h-5 w-3/4" />
      <Skeleton className="h-[46px] w-full rounded-3xl" />
      <div className="flex flex-col gap-2">
        {[0, 1, 2, 3, 4].map((i) => (
          <Skeleton key={i} className="h-16 w-full rounded-xl" />
        ))}
      </div>
    </div>
  );
}

interface ProviderRowProps {
  provider: ConnectableProvider;
  onSelect: (id: string) => void;
}

function ProviderRow({ provider, onSelect }: ProviderRowProps) {
  const [broken, setBroken] = useState(false);
  const src = `/integrations/${provider.id}.png`;

  return (
    <button
      type="button"
      onClick={() => onSelect(provider.id)}
      className="group flex h-16 w-full items-center gap-3 rounded-xl border border-zinc-200 bg-white px-[0.875rem] py-[0.625rem] text-left transition-colors hover:bg-zinc-50 active:bg-zinc-100 active:ring-1 active:ring-zinc-300 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400"
    >
      {broken ? (
        <div
          aria-hidden
          className="size-9 shrink-0 rounded-md bg-zinc-100"
        />
      ) : (
        <Image
          src={src}
          alt=""
          width={36}
          height={36}
          className="size-9 shrink-0 object-contain"
          onError={() => setBroken(true)}
          unoptimized
        />
      )}
      <span className="flex min-w-0 flex-1 flex-col gap-0.5">
        <span className="truncate text-[14px] font-medium leading-[22px] text-zinc-800">
          {provider.name}
        </span>
        <span className="truncate text-[12px] leading-[20px] text-zinc-500">
          {provider.description ?? provider.id}
        </span>
      </span>
      <span
        aria-hidden
        className="flex size-7 shrink-0 items-center justify-center rounded-lg bg-zinc-700 text-white transition-transform group-hover:bg-zinc-800 group-active:scale-[0.96]"
      >
        <PlusIcon size={18} weight="bold" />
      </span>
    </button>
  );
}
