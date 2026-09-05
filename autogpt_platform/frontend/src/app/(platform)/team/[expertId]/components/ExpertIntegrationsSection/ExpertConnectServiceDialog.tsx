"use client";

import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import { ConnectProviderRow } from "@/app/(platform)/copilot/components/OnboardingWelcomeDialog/ConnectProviderRow";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ConnectMethodView } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/components/ConnectMethodView/ConnectMethodView";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { cn } from "@/lib/utils";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useMeasuredHeight } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/useMeasuredHeight";
import { Plug01Icon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useRef } from "react";
import { useBottomScrollShadow } from "../../../components/SoulDrawer/useBottomScrollShadow";
import { useFitListToDialog } from "../useFitListToDialog";
import { useExpertConnectServiceDialog } from "./useExpertConnectServiceDialog";

const STEP_TRANSITION = { duration: 0.15, ease: [0, 0, 0.2, 1] as const };
const HEIGHT_TRANSITION = { duration: 0.2, ease: [0, 0, 0.2, 1] as const };

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

interface Props {
  open: boolean;
  expertName: string;
  onClose: () => void;
  onConnected: (credential: CredentialsMetaResponse) => void;
}

/** The welcome dialog's "Connect your tools" panel, minus the recommendations:
 *  every service is listed, and a successful connect hands the credential
 *  back so the expert is granted it. */
export function ExpertConnectServiceDialog({
  open,
  expertName,
  onClose,
  onConnected,
}: Props) {
  const {
    query,
    setQuery,
    providers,
    isLoading,
    isError,
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
    handleSuccess,
  } = useExpertConnectServiceDialog({ open, onConnected });
  const reduceMotion = useReducedMotion();
  const variants = reduceMotion ? reducedVariants : stepVariants;
  const [contentRef, contentHeight] = useMeasuredHeight<HTMLDivElement>();
  const listRef = useRef<HTMLUListElement | null>(null);
  const hasMoreBelow = useBottomScrollShadow(listRef);
  const attachList = useFitListToDialog(listRef);

  return (
    <Dialog
      styling={{ maxWidth: "40rem", maxHeight: "60vh" }}
      controlled={{
        isOpen: open,
        set: (next) => {
          if (!next && !isConnecting) onClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <motion.div
            className="relative overflow-hidden"
            animate={{ height: contentHeight ?? "auto" }}
            transition={reduceMotion ? { duration: 0 } : HEIGHT_TRANSITION}
          >
            <div ref={contentRef}>
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
                      onDeviceAuthSuccess={handleSuccess}
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
                    <div className="flex flex-col gap-1">
                      <Text
                        variant="h3"
                        className="!text-[1.25rem] text-zinc-900"
                      >
                        Connect a service for {expertName}
                      </Text>
                      <Text variant="small" className="!text-zinc-500">
                        Pick a service to connect. {expertName} will be able to
                        use it on your behalf.
                      </Text>
                    </div>
                    {isLoading ? (
                      <div className="grid grid-cols-2 gap-2">
                        {[0, 1, 2, 3].map((row) => (
                          <Skeleton
                            key={row}
                            className="h-14 w-full rounded-xl"
                          />
                        ))}
                      </div>
                    ) : isError ? (
                      <ErrorCard
                        context="services"
                        hint="We could not load the services you can connect."
                        onRetry={() => refetch()}
                      />
                    ) : (
                      <div className="flex flex-col gap-3">
                        <SearchInput
                          value={query}
                          onChange={setQuery}
                          placeholder="Search services..."
                          aria-label="Search services"
                        />
                        {providers.length === 0 ? (
                          <div className="flex flex-col items-center justify-center gap-2 rounded-2xl border border-dashed border-[#DADADC] py-8 text-center">
                            <Icon
                              icon={Plug01Icon}
                              size={24}
                              className="text-[#83838C]"
                            />
                            <Text variant="body" className="text-[#505057]">
                              {query.trim()
                                ? `No services match "${query.trim()}"`
                                : "No services available."}
                            </Text>
                          </div>
                        ) : (
                          <div className="relative">
                            <ul
                              ref={attachList}
                              className="grid grid-cols-2 gap-2 overflow-y-auto pr-1"
                              aria-label="Services"
                            >
                              {providers.map((provider) => (
                                <li key={provider.id}>
                                  <ConnectProviderRow
                                    provider={provider}
                                    onSelect={handleSelect}
                                    isConnected={connectedProviders.has(
                                      provider.id,
                                    )}
                                  />
                                </li>
                              ))}
                            </ul>

                            <div
                              aria-hidden="true"
                              className={cn(
                                "pointer-events-none absolute inset-x-0 bottom-0 h-10 bg-gradient-to-t from-white to-transparent transition-opacity duration-200",

                                hasMoreBelow ? "opacity-100" : "opacity-0",
                              )}
                            />
                          </div>
                        )}
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>

          <div className="flex items-center justify-end gap-3">
            {selectedProvider ? (
              <>
                <Button
                  variant="secondary"
                  size="small"
                  onClick={handleBackToList}
                >
                  Back
                </Button>
                {showContinue ? (
                  <Button
                    variant="primary"
                    size="small"
                    disabled={isContinueDisabled}
                    loading={isConnecting}
                    onClick={handleContinue}
                  >
                    {isConnecting ? "Connecting…" : "Continue"}
                  </Button>
                ) : null}
              </>
            ) : (
              <Button variant="secondary" size="small" onClick={onClose}>
                Cancel
              </Button>
            )}
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
