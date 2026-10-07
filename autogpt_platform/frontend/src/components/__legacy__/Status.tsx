import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import * as React from "react";

interface StatusProps {
  status: SubmissionStatus;
}

const statusConfig: Record<
  SubmissionStatus,
  {
    bgColor: string;
    dotColor: string;
    text: string;
  }
> = {
  [SubmissionStatus.DRAFT]: {
    bgColor: "bg-blue-50",
    dotColor: "bg-blue-500",
    text: "Draft",
  },
  [SubmissionStatus.PENDING]: {
    bgColor: "bg-yellow-50",
    dotColor: "bg-yellow-500",
    text: "Awaiting review",
  },
  [SubmissionStatus.APPROVED]: {
    bgColor: "bg-green-50",
    dotColor: "bg-green-500",
    text: "Approved",
  },
  [SubmissionStatus.REJECTED]: {
    bgColor: "bg-red-50",
    dotColor: "bg-red-500",
    text: "Rejected",
  },
};

export const Status: React.FC<StatusProps> = ({ status }) => {
  /**
   * Status component displays a badge with a colored dot and text indicating the agent's status
   * @param status - The current status of the agent
   *                 Valid values: 'draft', 'awaiting_review', 'approved', 'rejected'
   */
  if (!status) {
    return <Status status={SubmissionStatus.PENDING} />;
  } else if (!statusConfig[status]) {
    return <Status status={SubmissionStatus.PENDING} />;
  }

  const config = statusConfig[status];

  return (
    <div
      className={`px-2.5 py-1 ${config.bgColor} flex items-center gap-1.5 rounded-[26px]`}
    >
      <div className={`h-3 w-3 ${config.dotColor} rounded-full`} />
      <div className="font-sans text-sm leading-tight font-normal text-zinc-600">
        {config.text}
      </div>
    </div>
  );
};
