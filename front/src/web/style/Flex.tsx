import styled from "styled-components"

interface IFlex {
  direction?: "row" | "column"
  justify?:
    | "flex-start"
    | "flex-end"
    | "center"
    | "space-between"
    | "space-around"
  align?: "flex-start" | "flex-end" | "center" | "stretch" | "baseline"
  wrap?: "nowrap" | "wrap" | "wrap-reverse"
  gap?: number
  fullWidth?: boolean
  fullHeight?: boolean
}

const Flex = styled.div<IFlex>`
  display: flex;
  flex-direction: ${({ direction }) => direction || "row"};
  justify-content: ${({ justify }) => justify || "flex-start"};
  align-items: ${({ align }) => align || "stretch"};
  flex-wrap: ${({ wrap }) => wrap || "nowrap"};
  gap: ${({ gap }) => gap || 0}rem;
  width: ${({ fullWidth }) => (fullWidth ? "100%" : "auto")};
  height: ${({ fullHeight }) => (fullHeight ? "100%" : "auto")};
`

export default Flex
