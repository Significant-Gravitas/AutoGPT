import React, { useState } from "react";

import tw from "tailwind-styled-components";

import { TaskData } from "../../lib/types";
import RunButton from "./RunButton";
import MockCheckbox from "./MockCheckbox";

interface SelectedTaskProps {
  selectedTask: TaskData | null;
  isMock: boolean;
  setIsMock: React.Dispatch<React.SetStateAction<boolean>>;
  cutoff: number | null;
  setResponseData: React.Dispatch<React.SetStateAction<any>>;
  allResponseData: any[];
  setAllResponseData: React.Dispatch<React.SetStateAction<any[]>>;
}

const SelectedTask: React.FC<SelectedTaskProps> = ({
  selectedTask,
  isMock,
  setIsMock,
  cutoff,
  setResponseData,
  setAllResponseData,
  allResponseData,
}) => {
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const runTest = async () => {
    // If there's no selected task, do nothing
    if (!selectedTask?.name) return;

    const testParam = selectedTask.name;
    setIsLoading(true);
    try {
      let url = `http://localhost:8000/run_single_test?test=${testParam}&mock=${isMock}`;
      cutoff && !isMock && (url += `&cutoff=${cutoff}`);
      const response = await fetch(url);
      const data = await response.json();

      if (data["returncode"] > 0) {
        throw new Error(data["stderr"]);
      } else {
        const jsonObject = JSON.parse(data["stdout"]);
        setAllResponseData([...allResponseData, jsonObject]);
        setResponseData(jsonObject);
      }
    } catch (error) {
      console.error("There was an error fetching the data", error);
    }
    setIsLoading(false);
  };

  return (
    <>
      <TaskName>{selectedTask?.name}</TaskName>
      <TaskPrompt>{selectedTask?.task}</TaskPrompt>
      <Detail>
        <b>Cutoff:</b> {selectedTask?.cutoff}
      </Detail>
      <Detail>
        <b>Description:</b> {selectedTask?.info?.description}
      </Detail>
      <Detail>
        <b>Difficulty:</b> {selectedTask?.info?.difficulty}
      </Detail>
      <Detail>
        <b>Category:</b> {selectedTask?.category.join(", ")}
      </Detail>
      <RunButton
        cutoff={selectedTask?.cutoff}
        isLoading={isLoading}
        testRun={runTest}
        isMock={isMock}
      />
      <MockCheckbox isMock={isMock} setIsMock={setIsMock} />
    </>
  );
};

export default SelectedTask;

const TaskName = tw.h1`
    font-bold
    text-2xl
    break-words
`;

const TaskPrompt = tw.p`
    text-gray-900
    break-words
`;
const Detail = tw.p`
    mt-2
`;

const MockCheckboxInput = tw.input`
    border 
    rounded 
    focus:border-blue-400 
    focus:ring 
    focus:ring-blue-200 
    focus:ring-opacity-50
`;

const CheckboxWrapper = tw.label`
    flex 
    items-center 
    space-x-2 
    mt-2
`;
