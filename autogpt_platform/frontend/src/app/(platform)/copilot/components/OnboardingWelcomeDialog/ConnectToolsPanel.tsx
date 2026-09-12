"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ConnectMethodView } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/ConnectMethodView/ConnectMethodView";
import { ConnectProviderRow } from "./ConnectProviderRow";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Plug01Icon, Search01Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useEffect } from "react";
import { useConnectToolsPanel } from "./useConnectToolsPanel";
import { isKey } from "@/lib/keyboard";

interface Props {
  onBack: () => void;
  onNext: () => void;
}

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

const STEP_TRANSITION = { duration: 0.15, ease: [0, 0, 0.2, 1] as const };

// The provider picker embedded in the welcome dialog: same list → detail
// flow as ConnectServiceDialog, but connecting never closes the dialog —
// success slides back to the list so more tools can be wired up in one go.
export function ConnectToolsPanel({ onBack, onNext }: Props) {
  const {
    query,
    setQuery,
    providers,
    recommendedProviders,
    isPersonalized,
    isLoading,
    isError,
    error,
    refetch,
    selectedProvider,
    direction,
    connectedProviders,
    selectedMethod,
    setSelectedMethod,
    apiKeyForm,
    handleApiKeySubmit,
    showContinue,
    isContinueDisabled,
    isConnecting,
    handleSelect,
    handleBackToList,
    handleContinue,
  } = useConnectToolsPanel();
  const reduceMotion = useReducedMotion();
  const variants = reduceMotion ? reducedVariants : stepVariants;

  // Escape steps back one level (detail → list → cards) instead of
  // reaching the dialog's skip handler, which would end onboarding.
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (!isKey(event, "Escape")) return;
      if (selectedProvider) {
        handleBackToList();
        return;
      }
      onBack();
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedProvider, handleBackToList, onBack]);

  return (
    <div className="flex flex-col gap-4 px-5 pb-5 pt-4">
      <Text variant="h5" tone="primary">
        Connect your tools
      </Text>

      <div className="overflow-hidden">
        <AnimatePresence mode="wait" initial={false} custom={direction}>
          {selectedProvider ? (
            <motion.div
              key={`detail-${selectedProvider.id}`}
              custom={direction}
              variants={variants}
              initial="initial"
              animate="active"
              exit="exit"
              transition={STEP_TRANSITION}
            >
              <ConnectMethodView
                provider={selectedProvider}
                selectedMethod={selectedMethod}
                onSelectMethod={setSelectedMethod}
                apiKeyForm={apiKeyForm}
                onApiKeySubmit={handleApiKeySubmit}
                // Without this, approving on the phone drops the user back on
                // the initial "Connect <provider>" screen, which reads as a
                // failure. OAuth and API key both return to the list.
                onDeviceAuthSuccess={handleBackToList}
              />
            </motion.div>
          ) : (
            <motion.div
              key="list"
              custom={direction}
              variants={variants}
              initial="initial"
              animate="active"
              exit="exit"
              transition={STEP_TRANSITION}
              className="flex flex-col gap-4"
            >
              {isLoading ? (
                <div className="flex flex-col gap-2">
                  {[0, 1, 2].map((row) => (
                    <Skeleton key={row} className="h-14 w-full rounded-md" />
                  ))}
                </div>
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
                <div className="flex flex-col gap-4">
                  <div className="relative w-full">
                    <Icon
                      icon={Search01Icon}
                      size={16}
                      className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-zinc-400"
                    />
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Search services..."
                      aria-label="Search services"
                      className="h-9 w-full rounded-md border border-zinc-200 bg-white pl-9 pr-3 text-sm leading-[22px] text-zinc-900 transition-colors placeholder:text-zinc-400 focus:border-zinc-900 focus:outline-none"
                    />
                  </div>

                  {query.trim() ? (
                    providers.length === 0 ? (
                      <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-zinc-200 py-8 text-center">
                        <Icon
                          icon={Plug01Icon}
                          size={16}
                          className="text-zinc-400"
                        />
                        <Text variant="body" tone="secondary">
                          {`No services match "${query.trim()}"`}
                        </Text>
                      </div>
                    ) : (
                      <ul className="grid max-h-[13.5rem] grid-cols-2 gap-2 overflow-y-auto pr-1">
                        {providers.map((provider) => (
                          <li key={provider.id}>
                            <ConnectProviderRow
                              provider={provider}
                              onSelect={handleSelect}
                              isConnected={connectedProviders.has(provider.id)}
                            />
                          </li>
                        ))}
                      </ul>
                    )
                  ) : recommendedProviders.length > 0 ? (
                    <div className="flex flex-col gap-2">
                      <Text variant="eyebrow" as="span">
                        {isPersonalized
                          ? "Recommended from our conversation"
                          : "Popular places to start"}
                      </Text>
                      <ul className="grid grid-cols-2 gap-2">
                        {recommendedProviders.map((provider) => (
                          <li key={provider.id}>
                            <ConnectProviderRow
                              provider={provider}
                              onSelect={handleSelect}
                              isConnected={connectedProviders.has(provider.id)}
                              description={provider.description}
                            />
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : (
                    <Text
                      variant="body"
                      tone="secondary"
                      className="text-center"
                    >
                      Search to find a service to connect.
                    </Text>
                  )}
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="flex items-center justify-end gap-2">
        {selectedProvider ? (
          <>
            <Button variant="outline" size="xs" onClick={handleBackToList}>
              Back
            </Button>
            {showContinue && (
              <Button
                variant="primary"
                size="xs"
                disabled={isContinueDisabled}
                loading={isConnecting}
                onClick={handleContinue}
              >
                {isConnecting ? "Connecting…" : "Continue"}
              </Button>
            )}
          </>
        ) : (
          <>
            <Button variant="outline" size="xs" onClick={onBack}>
              Back
            </Button>
            <Button variant="primary" size="xs" onClick={onNext}>
              Next
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
