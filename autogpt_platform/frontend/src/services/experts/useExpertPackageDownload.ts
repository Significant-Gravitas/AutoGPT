import {
  downloadExpertPackage,
  downloadExpertTemplatePackage,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  downloadFile,
  filenameFromContentDisposition,
} from "@/lib/download-file";
import { useState } from "react";
import {
  trackExpertDownloaded,
  trackExpertExported,
} from "./experts-analytics";
import { expertPackageFilename } from "./helpers";

interface Args {
  /** An owned expert exports itself; a template has its own public route. */
  kind: "expert" | "template";
  id: string;
  name: string;
  workflowCount: number;
  skillCount: number;
}

export function useExpertPackageDownload({
  kind,
  id,
  name,
  workflowCount,
  skillCount,
}: Args) {
  const { toast } = useToast();
  const [isDownloading, setIsDownloading] = useState(false);

  async function download() {
    setIsDownloading(true);
    try {
      const response =
        kind === "expert"
          ? await downloadExpertPackage(id)
          : await downloadExpertTemplatePackage(id);

      if (response.status !== 200) {
        throw new Error(`Failed to download expert (HTTP ${response.status})`);
      }

      downloadFile(
        filenameFromContentDisposition(
          response.headers,
          expertPackageFilename(name),
        ),
        response.data,
      );

      const payload = {
        expert_id: id,
        workflow_count: workflowCount,
        skill_count: skillCount,
      };
      if (kind === "expert") {
        trackExpertExported(payload);
      } else {
        trackExpertDownloaded(payload);
      }
    } catch (error) {
      toast({
        title: "Couldn't download this expert",
        description:
          error instanceof Error
            ? error.message
            : "An unexpected error occurred.",
        variant: "destructive",
      });
    } finally {
      setIsDownloading(false);
    }
  }

  return { isDownloading, download };
}
