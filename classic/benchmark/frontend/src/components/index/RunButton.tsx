import React, { useState, useEffect } from "react";

import tw from "tailwind-styled-components";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCircleNotch } from "@fortawesome/free-solid-svg-icons";

interface RunButtonProps {
  testRun: () => Promise<void>;
  isLoading: boolean;
  cutoff?: string;
  isMock: boolean;
}

const RunButton: React.FC<RunButtonProps> = ({
  testRun,
  isLoading,
  cutoff,
  isMock,
}) => {
  const intCutoff = cutoff ? parseInt(cutoff) : null;
  const [timeElapsed, setTimeElapsed] = useState<number>(0);

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;

    if (isLoading) {
      interval = setInterval(() => {
        setTimeElapsed((prevTime) => prevTime + 1);
      }, 1000);
    } else {
      if (interval !== null) {
        clearInterval(interval);
      }
      setTimeElapsed(0); // Reset the timer when not loading
    }

    return () => {
      if (interval !== null) {
        clearInterval(interval);
      }
    };
  }, [isLoading]);

  const timeUntilCutoff = intCutoff ? intCutoff - timeElapsed : null;

  return (
    <>
      <RunButtonWrapper onClick={testRun}>
        {!isLoading ? (
          "Run Task"
        ) : (
          <FontAwesomeIcon size="lg" icon={faCircleNotch} spin />
        )}
      </RunButtonWrapper>
      {cutoff && isLoading && (
        <>
          {isMock ? (
            <p>Time elapsed: {timeElapsed} seconds</p>
          ) : (
            <p>Time until cutoff: {timeUntilCutoff} seconds</p>
          )}
        </>
      )}
    </>
  );
};

export default RunButton;

const RunButtonWrapper = tw.button`
    border
    mt-4
    py-1
    px-3
    w-28
    rounded
    flex
    items-center
    justify-center
`;
