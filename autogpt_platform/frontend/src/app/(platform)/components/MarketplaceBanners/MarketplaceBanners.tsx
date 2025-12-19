"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface MarketplaceBannersProps {
  hasUpdate?: boolean;
  latestVersion?: number;
  hasUnpublishedChanges?: boolean;
  currentVersion?: number;
  isUpdating?: boolean;
  onUpdate?: () => void;
  onPublish?: () => void;
  onViewChanges?: () => void;
}

export function MarketplaceBanners({
  hasUpdate,
  latestVersion,
  hasUnpublishedChanges,
  isUpdating,
  onUpdate,
  onPublish,
}: MarketplaceBannersProps) {
  const renderUpdateBanner = () => {
    if (hasUpdate && latestVersion) {
      return (
        <div className="mb-6 rounded-lg bg-gray-50 p-4 dark:bg-gray-900">
          <div className="flex flex-col gap-3">
            <Text variant="body" className="text-gray-700 dark:text-gray-300">
              You should update your agent in order to get the latest / best
              results
            </Text>
            {onUpdate && (
              <div className="flex justify-start">
                <Button
                  size="small"
                  onClick={onUpdate}
                  disabled={isUpdating}
                  className="bg-gray-700 text-white hover:bg-gray-800 dark:bg-gray-600 dark:hover:bg-gray-700"
                >
                  {isUpdating ? "Updating..." : "Update agent"}
                </Button>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  const renderUnpublishedChangesBanner = () => {
    if (hasUnpublishedChanges) {
      return (
        <div className="mb-6 rounded-lg bg-gray-50 p-4 dark:bg-gray-900">
          <div className="flex flex-col gap-3">
            <Text variant="body" className="text-gray-700 dark:text-gray-300">
              You&apos;ve made changes to this agent that aren&apos;t published
              yet. Would you like to publish the latest version?
            </Text>
            {onPublish && (
              <div className="flex justify-start">
                <Button
                  size="small"
                  onClick={onPublish}
                  className="bg-gray-700 text-white hover:bg-gray-800 dark:bg-gray-600 dark:hover:bg-gray-700"
                >
                  Publish changes
                </Button>
              </div>
            )}
          </div>
        </div>
      );
    }
    return null;
  };

  return (
    <>
      {renderUpdateBanner()}
      {renderUnpublishedChangesBanner()}
    </>
  );
}
