import React, { useState, useEffect } from "react";
import tw from "tailwind-styled-components";

import Dashboard from "~/components/data/Dashboard";
import Reports from "~/components/data/Reports";

const DataPage: React.FC = () => {
  const [data, setData] = useState<any>([]);
  const getData = async () => {
    try {
      let url = `http://localhost:8000/data`;
      const response = await fetch(url);
      const responseData = await response.json();

      setData(responseData);
    } catch (error) {
      console.error("There was an error fetching the data", error);
    }
  };

  useEffect(() => {
    getData();
  }, []);

  return (
    <PageContainer>
      <Dashboard data={data} />
      <Reports data={data} />
    </PageContainer>
  );
};

export default DataPage;

const PageContainer = tw.div`
  px-12
  w-full
  h-full
  min-h-screen
  bg-gray-50
`;
