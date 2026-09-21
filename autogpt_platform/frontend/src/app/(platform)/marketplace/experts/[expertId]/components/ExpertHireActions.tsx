import { Expert } from "@/app/api/__generated__/models/expert";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { trackFunnel } from "@/services/experts/experts-analytics";
import { markHireStarted } from "@/services/experts/hire-timing";
import { useExpertPackageDownload } from "@/services/experts/useExpertPackageDownload";
import {
  CheckmarkCircle02Icon,
  Download01Icon,
} from "@hugeicons/core-free-icons";

// Sized and shaped like the small button beside it so the pair reads as one row.
const STATUS_CLASS = "h-9 rounded-full px-3.5 text-sm";

// A white, hairline-bordered secondary so the button sits quietly next to
// the status badge instead of reading as a solid grey slab.
const SECONDARY_CLASS =
  "border-zinc-200 bg-white shadow-[0_1px_2px_rgba(16,24,40,0.05)] hover:border-zinc-300 hover:bg-zinc-50";

interface Props {
  expert: Expert;
  hiredExpert: Expert | null;
  isLoggedIn: boolean;
  canDownload: boolean;
  isHiring: boolean;
  onHire: () => void;
}

/** The page's one call to action, sitting in the header beside the name.
 *  Signed-out visitors get a sign-up prompt that brings them back here. */
export function ExpertHireActions({
  expert,
  hiredExpert,
  isLoggedIn,
  canDownload,
  isHiring,
  onHire,
}: Props) {
  // The page's id is the template's: a marketplace download always reads the
  // published template, never the viewer's own copy of it.
  const { isDownloading, download } = useExpertPackageDownload({
    kind: "template",
    id: expert.id,
    name: expert.name,
    workflowCount: expert.workflows.length,
    skillCount: expert.skills.length,
  });

  // Icon-only: the atom shows the aria-label as a hover tooltip.
  const downloadButton = !canDownload ? null : (
    <Button
      variant="icon"
      size="icon"
      aria-label="Download as file"
      loading={isDownloading}
      onClick={download}
      data-testid="expert-export-button"
    >
      <Icon icon={Download01Icon} size={16} />
    </Button>
  );

  if (!isLoggedIn) {
    const next = encodeURIComponent(`/marketplace/experts/${expert.id}`);
    return (
      <Button
        as="NextLink"
        href={`/signup?next=${next}`}
        variant="primary"
        size="small"
        className="w-full sm:w-auto"
      >
        Get started
      </Button>
    );
  }

  if (hiredExpert) {
    return (
      <div className="flex flex-wrap items-center gap-3">
        <Badge variant="success" className={STATUS_CLASS}>
          <Icon icon={CheckmarkCircle02Icon} size={16} />
          On your team
        </Badge>
        {downloadButton}
        <Button
          as="NextLink"
          href={`/copilot?expertId=${hiredExpert.id}`}
          variant="secondary"
          size="small"
          className={SECONDARY_CLASS}
        >
          {`Chat with ${expert.name}`}
        </Button>
      </div>
    );
  }

  // The clock starts on the click, not on the request: the hire flow finishes
  // in a dialog, and sometimes on another page entirely.
  function handleHire() {
    markHireStarted(expert.id);
    trackFunnel("hire_started", { template_id: expert.id });
    onHire();
  }

  return (
    <div className="flex items-center gap-2">
      {downloadButton}
      <Button
        variant="primary"
        size="small"
        onClick={handleHire}
        loading={isHiring}
        className="w-full sm:w-auto"
      >
        {`Hire ${expert.name}`}
      </Button>
    </div>
  );
}
