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
export const OnOff = styled.div`
  color: var(--grey100);
  font-size: 1rem;
  font-weight: 500;
`
export const Grid = styled.div`
  display: grid;
  grid-template-columns: var(--ai-list-width) 1fr;
  width: 100%;
  `