import * as React from "react";
import { Button } from "./Button";

interface BecomeACreatorProps {
  title: string;
  heading: string;
  description: string;
  buttonText: string;
  onButtonClick: () => void;
}

export const BecomeACreator: React.FC<BecomeACreatorProps> = ({
  title = "Want to contribute?",
  heading = "We're always looking for more Creators!",
  description = "Join our ever-growing community of hackers and tinkerers",
  buttonText = "Become a Creator",
  onButtonClick = () => {},
}) => {
  return (
    <div className="flex w-full flex-col items-center justify-between space-y-8 py-8">
      <div className="font-neue mb-8 self-start text-[23px] font-bold leading-9 tracking-tight text-[#282828]">
        {title}
      </div>
      <div className="font-neue max-w-full text-center text-5xl font-medium leading-9 tracking-wide text-[#272727]">
        {heading}
      </div>
      <div className="font-neue max-w-full text-center text-[26px] font-medium leading-9 tracking-tight text-[#878787]">
        {description}
      </div>
      <Button onClick={onButtonClick} className="mt-8">
        {buttonText}
      </Button>
    </div>
  );
};
