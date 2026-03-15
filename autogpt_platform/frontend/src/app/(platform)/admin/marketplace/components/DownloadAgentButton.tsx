"use client";

import { downloadAsAdmin } from "@/app/(platform)/admin/marketplace/actions";
import { Button } from "@/components/__legacy__/ui/button";
import { ExternalLink } from "lucide-react";
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

      // Client-side download logic
      const jsonData = JSON.stringify(fileData, null, 2);
      const blob = new Blob([jsonData], { type: "application/json" });

      // Create a temporary URL for the Blob
      const url = window.URL.createObjectURL(blob);

      // Create a temporary anchor element
      const a = document.createElement("a");
      a.href = url;
      a.download = `agent_${storeListingVersionId}.json`;

      // Append the anchor to the body, click it, and remove it
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);

      // Revoke the temporary URL
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Download failed:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Button
      size="sm"
      variant="outline"
      onClick={handleDownload}
      disabled={isLoading}
    >
      <ExternalLink className="mr-2 h-4 w-4" />
      {isLoading ? "Downloading..." : "Download"}
    </Button>
  );
}
