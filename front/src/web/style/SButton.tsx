import React from "react"
import { useRef } from "react"
import styled, { css, keyframes } from "styled-components"
import colored from "./colored"
import { Button } from "@mui/material"

const StyledButton = styled(Button)<{
  variant?: "contained" | "outlined" | "text"
  $noScale?: boolean
}>`
  padding: 0.5rem 1rem;
  border-radius: 0.5rem;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  outline: none;
  border: none;
  text-transform: none;

  ${({ variant }) =>
    (variant === undefined || variant === "contained") &&
    css`
      box-shadow: var(--shadow-elevation-low) !important;
      background-color: var(--color) !important;
      color: var(--text-color) !important;
      &:hover {
        box-shadow: var(--shadow-elevation-medium) !important;
        background-color: var(--color-hover) !important;
      }
      &:active {
        box-shadow: none !important;
        background-color: var(--color-active) !important;
      }
    `}

  ${({ variant }) =>
    variant === "outlined" &&
    css`
      background-color: inherit !important;
      box-shadow: none !important;
      border: 1px solid var(--color) !important;
      color: var(--color) !important;
      --text-color: var(--color) !important;
      &:hover {
        border: 1px solid var(--color-hover) !important;
      }
      &:active {
        border: 1px solid var(--color-active) !important;
      }
    `}

      ${({ variant }) =>
    variant === "text" &&
    css`
      box-shadow: none !important;
      color: var(--color) !important;
      --text-color: var(--color);
      &:hover {
        color: var(--color-hover) !important;
      }
      &:active {
        color: var(--color-active) !important;
      }
    `}
  
  ${({ $noScale = false }) =>
    !$noScale &&
    css`
      &:hover {
        transition: transform 0.2s !important;
        transform: scale(1.05) !important;
      }
      &:active {
        transition: transform 0.2s !important;
        transform: scale(0.95) !important;
      }
    `}
  ${({ disabled }) =>
    disabled &&
    css`
      box-shadow: none !important;
      opacity: 0.5 !important;
      cursor: not-allowed !important;
      pointer-events: all !important;
      transform: none !important;
      &:hover {
        transform: none !important;
      }
      &:active {
        transform: none !important;
      }
    `}
`
const SButton = colored(StyledButton)
export default SButton
