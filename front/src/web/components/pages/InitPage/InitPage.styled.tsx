import { IconButton, Switch } from "@mui/material"
import styled from "styled-components"

export const Container = styled.div`
  background-color: var(--grey800);
  // full page center on screen
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  width: 100%;
  align-self: center;
`

export const ColoredIconButton = styled(IconButton)`
  color: var(--grey100) !important;
`

export const Modal = styled.div`
  max-width: 800px;
  width: 100%;
`
export const Label = styled.div`
  color: var(--grey100);
`
export const StyledSwitch = styled(Switch)<{ $active?: boolean }>`
  & * {
    color: var(--yellow) !important;
  }
  & .MuiSwitch-track {
    background-color: var(--yellow100) !important;
    ${({ $active }) =>
      $active &&
      `
      background-color: var(--yellow) !important;
    `}
`
