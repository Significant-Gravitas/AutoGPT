"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { COOKIE_CATEGORIES } from "@/services/consent/cookies";
import { CheckIcon } from "@phosphor-icons/react/dist/ssr";
import { useCookieSettingsModal } from "./useCookieSettingsModal";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

export function CookieSettingsModal({ isOpen, onClose }: Props) {
  const {
    analytics,
    setAnalytics,
    monitoring,
    setMonitoring,
    handleSavePreferences,
    handleAcceptAll,
    handleRejectAll,
  } = useCookieSettingsModal({ onClose });

  return (
    <Dialog
      title="Cookie Settings"
      controlled={{
        isOpen,
        set: (open) => {
          if (!open) onClose();
        },
      }}
    >
      <Dialog.Content>
        <div className="space-y-6 pb-6">
          <div className="space-y-2">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 space-y-1">
                <div className="flex items-center gap-2">
                  <Text
                    variant="body-medium"
                    className="text-neutral-900 dark:text-neutral-100"
                  >
                    {COOKIE_CATEGORIES.essential.name}
                  </Text>
                  <span className="rounded-full bg-neutral-100 px-2 py-0.5 text-xs font-medium text-neutral-600 dark:bg-neutral-800 dark:text-neutral-400">
                    Always Active
                  </span>
                </div>
                <Text
                  variant="body"
                  className="text-neutral-600 dark:text-neutral-400"
                >
                  {COOKIE_CATEGORIES.essential.description}
                </Text>
              </div>
              <div className="flex items-center">
                <CheckIcon className="h-5 w-5 text-green-600" weight="bold" />
              </div>
            </div>
          </div>

          <div className="border-t border-neutral-200 dark:border-neutral-800" />

          <div className="space-y-2">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 space-y-1">
                <Text
                  variant="body-medium"
                  className="text-neutral-900 dark:text-neutral-100"
                >
                  {COOKIE_CATEGORIES.analytics.name}
                </Text>
                <Text
                  variant="body"
                  className="text-neutral-600 dark:text-neutral-400"
                >
                  {COOKIE_CATEGORIES.analytics.description}
                </Text>
              </div>
              <Switch
                checked={analytics}
                onCheckedChange={setAnalytics}
                aria-label="Toggle analytics cookies"
              />
            </div>
          </div>

          <div className="border-t border-neutral-200 dark:border-neutral-800" />

          <div className="space-y-2">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 space-y-1">
                <Text
                  variant="body-medium"
                  className="text-neutral-900 dark:text-neutral-100"
                >
                  {COOKIE_CATEGORIES.monitoring.name}
                </Text>
                <Text
                  variant="body"
                  className="text-neutral-600 dark:text-neutral-400"
                >
                  {COOKIE_CATEGORIES.monitoring.description}
                </Text>
              </div>
              <Switch
                checked={monitoring}
                onCheckedChange={setMonitoring}
                aria-label="Toggle monitoring cookies"
              />
            </div>
          </div>
        </div>

        <Dialog.Footer>
          <div className="flex w-full flex-col-reverse gap-2 sm:flex-row sm:justify-end">
            <Button variant="ghost" size="small" onClick={handleRejectAll}>
              Reject All
            </Button>
            <Button variant="ghost" size="small" onClick={handleAcceptAll}>
              Accept All
            </Button>
            <Button
              variant="primary"
              size="small"
              onClick={handleSavePreferences}
            >
              Save Preferences
            </Button>
          </div>
        </Dialog.Footer>
      </Dialog.Content>
    </Dialog>
  );
}
