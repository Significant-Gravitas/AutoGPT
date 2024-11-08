import * as React from "react";

export type StatusType = "draft" | "awaiting_review" | "approved" | "rejected";

interface StatusProps {
  status: StatusType;
}

const statusConfig: Record<StatusType, {
  bgColor: string;
  dotColor: string;
  text: string;
}> = {
  draft: {
    bgColor: "bg-blue-50",
    dotColor: "bg-blue-500",
    text: "Draft"
  },
  awaiting_review: {
    bgColor: "bg-amber-50",
    dotColor: "bg-amber-500",
    text: "Awaiting review"
  },
  approved: {
    bgColor: "bg-green-50",
    dotColor: "bg-green-500",
    text: "Approved"
  },
  rejected: {
    bgColor: "bg-red-50",
    dotColor: "bg-red-500",
    text: "Rejected"
  }
};

export const Status: React.FC<StatusProps> = ({ status }) => {
  if (!status || !statusConfig[status]) {
    return null;
  }

  const config = statusConfig[status];
  
  return (
    <div className={`px-2.5 py-1 ${config.bgColor} rounded-[26px] flex items-center gap-1.5`}>
      <div className={`w-3 h-3 ${config.dotColor} rounded-full`} />
      <div className="text-neutral-600 text-sm font-normal font-['Geist'] leading-tight">
        {config.text}
      </div>
    </div>
  );
};