import React from "react";

import tw from "tailwind-styled-components";

interface MockCheckboxProps {
  isMock: boolean;
  setIsMock: React.Dispatch<React.SetStateAction<boolean>>;
}

const MockCheckbox: React.FC<MockCheckboxProps> = ({ isMock, setIsMock }) => {
  return (
    <CheckboxWrapper>
      <MockCheckboxInput
        type="checkbox"
        checked={isMock}
        onChange={() => setIsMock(!isMock)}
      />
      <span>Run mock test</span>
    </CheckboxWrapper>
  );
};

export default MockCheckbox;

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
