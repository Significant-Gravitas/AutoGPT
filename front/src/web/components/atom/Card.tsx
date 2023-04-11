import { Paper } from "@mui/material"
import styled from "styled-components"

export const Card = styled(Paper)<{ $active?: boolean }>`
  color: var(--grey100) !important;
  padding: 1rem;
  border-radius: 0.5rem;
  background-color: var(--grey900) !important;
  ${({ $active }) => $active && "background-color: var(--grey800) !important;"}
  &:hover {
    background-color: var(--grey700) !important;
    cursor: pointer;
  }
  & > p {
    margin: 0;
    font-size: 0.8rem;
    color: var(--grey300);
  }
`
