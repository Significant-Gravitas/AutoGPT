"use client";

import {
  AnimatePresence,
  MotionConfig,
  motion,
  useReducedMotion,
} from "framer-motion";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { ListLoading, ListView } from "./components/ListView";
import { DetailView } from "./components/DetailView/DetailView";
import { useConnectServiceDialog } from "./useConnectServiceDialog";
import { useMeasuredHeight } from "./useMeasuredHeight";

interface Props {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const TRANSITION = { duration: 0.15, ease: [0, 0, 0.2, 1] as const };
const HEIGHT_TRANSITION = {
  duration: 0.2,
  ease: [0, 0, 0.2, 1] as const,
};

const stepVariants = {
  initial: (direction: number) => ({ x: 16 * direction, opacity: 0 }),
  active: { x: 0, opacity: 1 },
  exit: (direction: number) => ({ x: -16 * direction, opacity: 0 }),
};

const reducedVariants = {
  initial: { opacity: 0 },
  active: { opacity: 1 },
  exit: { opacity: 0 },
};

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
          >
            <div ref={contentRef}>
              <AnimatePresence mode="wait" initial={false} custom={direction}>
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
