import React, { useState } from "react";
import { LatestRun } from "../../lib/types";
import tw from "tailwind-styled-components";

const RecursiveDropdown: React.FC<{ data: any; skipKeys: string[] }> = ({
  data,
  skipKeys,
}) => {
  if (typeof data !== "object" || data === null) {
    return null;
  }

  return (
    <>
      {Object.entries(data).map(([key, value]) => {
        if (skipKeys.includes(key)) {
          return null;
        }

        // Special case for 'category' key
        if (key === "category" && Array.isArray(value)) {
          return (
            <Section key={key}>
              <Label>{key}:</Label>
              <Data>{value.join(", ")}</Data>
            </Section>
          );
        }

        if (typeof value === "object" && value !== null) {
          return (
            <Dropdown key={key}>
              <DropdownSummary>{key}</DropdownSummary>
              <DropdownContent>
                <RecursiveDropdown data={value} skipKeys={skipKeys} />
              </DropdownContent>
            </Dropdown>
          );
        } else {
          return (
            <Section key={key}>
              <Label>{key}:</Label>
              <Data>
                {typeof value === "string" ? value : JSON.stringify(value)}
              </Data>
            </Section>
          );
        }
      })}
    </>
  );
};

const RunData: React.FC<{ latestRun: LatestRun }> = ({ latestRun }) => {
  const date = new Date(latestRun.benchmark_start_time);
  return (
    <Card>
      <Section>
        <Label>Command:</Label>
        <Data>{latestRun.command}</Data>
      </Section>
      <Section>
        <Label>Start time:</Label>
        <Data>{date.toLocaleString()}</Data>
      </Section>
      <Section>
        <Label>Run time:</Label>
        <Data>{latestRun.metrics.run_time}</Data>
      </Section>
      <Section>
        <Label>Highest difficulty:</Label>
        <Data>
          {latestRun.metrics.highest_difficulty.split(":")[1]?.slice(-1)}
        </Data>
      </Section>

      {Object.keys(latestRun.tests).map((testKey) => (
        <Dropdown key={testKey}>
          <DropdownSummary>{testKey}</DropdownSummary>
          <DropdownContent>
            {latestRun.tests[testKey] && (
              <RecursiveDropdown
                data={latestRun.tests[testKey]}
                skipKeys={["cost", "data_path"]}
              />
            )}
          </DropdownContent>
        </Dropdown>
      ))}
    </Card>
  );
};

export default RunData;

const Card = tw.div`
  bg-white
  p-4
  rounded
  shadow-lg
  w-full
  mt-4
`;

const Section = tw.div`
  mt-2
`;

const Label = tw.span`
  font-medium
`;

const Data = tw.span`
  ml-1
`;

const Dropdown = tw.details`
  mt-4
`;

const DropdownSummary = tw.summary`
  cursor-pointer
  text-blue-500
`;

const DropdownContent = tw.div`
  pl-4
  mt-2
`;
