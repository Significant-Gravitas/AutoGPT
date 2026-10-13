import { OpenUI } from "@/components/organisms/OpenUI/OpenUI";
import { FileViewHeader, type FileViewHeaderProps } from "./FileViewHeader";

const STALE_NOTICE =
  "A newer version of this view is saved in your files. This older version is read-only.";

interface Props {
  source: string;
  header?: FileViewHeaderProps;
  notice?: string;
}

export function StaleRevision({
  source,
  header,
  notice = STALE_NOTICE,
}: Props) {
  return (
    <section
      aria-label="Interactive response"
      className="my-3 min-w-0 space-y-3"
    >
      {header && <FileViewHeader {...header} />}
      <p role="status" className="text-xs text-muted-foreground">
        {notice}
      </p>
      <OpenUI
        source={source}
        isStreaming={false}
        disabled
        revision={0}
        onAction={() => {}}
      />
    </section>
  );
}
