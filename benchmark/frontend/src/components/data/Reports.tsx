import React, { useState } from "react";
import tw from "tailwind-styled-components";

interface ReportsProps {
  data: any;
}

const Reports: React.FC<ReportsProps> = ({ data }) => {
  return (
    <ReportsContainer>
      <Table></Table>
    </ReportsContainer>
  );
};

export default Reports;

const ReportsContainer = tw.div`
  w-full
`;

const Table = tw.div`
  w-full
  border
  shadow-lg
  rounded-xl
  h-96
`;
