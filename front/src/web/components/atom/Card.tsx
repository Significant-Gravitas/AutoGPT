import colored from "@/style/colored"
import { Paper } from "@mui/material"
import styled from "styled-components"

export const Card = colored(styled(Paper)<{
  $active?: boolean
  $noPadding?: boolean
  $fitContent?: boolean
}>`
  padding: 1rem;
  border-radius: 0.5rem;
  background-color: var(--color) !important;
  border-color: var(--border-color) !important;
  border: 1px solid;
  ${({ $fitContent }) => $fitContent && "width: fit-content;"}
  ${({ $active }) => $active && "background-color: var(--grey800) !important;"}
  ${({ $noPadding }) => $noPadding && "padding: 0;"}
  &:hover {
    background-color: var(--grey700) !important;
    cursor: pointer;
  }
  & > * {
    margin: 0;
    font-size: 0.8rem;
    color: var(--text-color);
  }
`)
