import React, { useState } from "react";
import tw from "tailwind-styled-components";

interface CategorySuccessProps {
  data: any;
}

const CategorySuccess: React.FC<CategorySuccessProps> = ({ data }) => {
  return <CategorySuccessContainer></CategorySuccessContainer>;
};

export default CategorySuccess;

const CategorySuccessContainer = tw.div`
  
`;
