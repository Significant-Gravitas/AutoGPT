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
    <div className="flex w-full flex-col items-center justify-between space-y-4 py-8 leading-9 md:space-y-8">
      <div className="mb:mb-8 mb-4 self-start font-neue text-xl font-bold tracking-tight text-[#282828] md:text-[23px]">
        {title}
      </div>
      <div className="max-w-full text-center font-neue text-4xl font-medium tracking-wide text-[#272727] md:text-5xl">
        {heading}
      </div>
      <div className="max-w-full text-center font-neue text-xl font-medium tracking-tight text-[#737373] md:text-[26px]">
        {description}
      </div>
      <Button onClick={onButtonClick} className="mt-8">
        {buttonText}
      </Button>
    </div>
  );
};
