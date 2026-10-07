"use client";

import { downloadAsAdmin } from "@/app/(platform)/admin/marketplace/actions";
import { Button } from "@/components/atoms/Button/Button";
import { agentGraphExportFilename, exportAsJSONFile } from "@/lib/utils";
import { LinkSquare02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";

export function DownloadAgentAdminButton({
  storeListingVersionId,
}: {
  storeListingVersionId: string;
}) {
  const [isLoading, setIsLoading] = useState(false);

  const handleDownload = async () => {
    try {
      setIsLoading(true);
      // Call the server action to get the data
      const fileData = await downloadAsAdmin(storeListingVersionId);

      exportAsJSONFile(fileData as object, agentGraphExportFilename(fileData));
    } catch (error) {
      console.error("Download failed:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Button
      size="md"
      variant="outline"
      onClick={handleDownload}
      disabled={isLoading}
      leadingIcon={LinkSquare02Icon}
    >
      {isLoading ? "Downloading..." : "Download"}
    </Button>
  );
}
