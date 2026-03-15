"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CookieIcon } from "@phosphor-icons/react/dist/ssr";
import { useCookieConsentBanner } from "./useCookieConsentBanner";
import { CookieSettingsModal } from "./components/CookieSettingsModal/CookieSettingsModal";

export function CookieConsentBanner() {
  const {
    shouldShowBanner,
    showSettings,
    handleAcceptAll,
    handleRejectAll,
    handleOpenSettings,
    handleCloseSettings,
  } = useCookieConsentBanner();

  if (!shouldShowBanner) {
    return null;
  }

  return (
    <>
      <div className="fixed bottom-0 left-0 right-0 z-50 px-10 pb-4">
        <div
          className="mx-auto max-w-6xl rounded-lg border border-neutral-200 bg-white p-4 shadow-lg dark:border-neutral-800 dark:bg-neutral-950"
          role="dialog"
          aria-label="Cookie consent banner"
        >
          <div className="flex flex-col items-start gap-4 md:flex-row md:items-center md:justify-between">
            <div className="flex flex-1 items-start gap-3">
              <CookieIcon className="mt-0.5 h-5 w-5 shrink-0 text-neutral-700 dark:text-neutral-300" />
              <div className="flex-1">
                <Text
                  variant="body-medium"
                  className="mb-1 text-neutral-900 dark:text-neutral-100"
                >
                  We use cookies
                </Text>
                <Text
                  variant="body"
                  className="text-neutral-600 dark:text-neutral-400"
                >
                  AutoGPT uses essential cookies for login and optional cookies
                  for analytics and error tracking to improve our service.
                </Text>
              </div>
            </div>

            <div className="flex w-full flex-col gap-2 md:w-auto md:flex-row md:items-center">
              <Button
                variant="ghost"
                size="small"
                onClick={handleRejectAll}
                className="w-full md:w-auto"
              >
                Reject All
              </Button>
              <Button
                variant="ghost"
                size="small"
                onClick={handleOpenSettings}
                className="w-full md:w-auto"
              >
                Settings
              </Button>
              <Button
                variant="primary"
                size="small"
                onClick={handleAcceptAll}
                className="w-full md:w-auto"
              >
                Accept All
              </Button>
            </div>
          </div>
        </div>
      </div>

      {showSettings && (
        <CookieSettingsModal
          isOpen={showSettings}
          onClose={handleCloseSettings}
        />
      )}
    </>
  );
}
