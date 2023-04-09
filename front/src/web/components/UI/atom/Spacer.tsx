import styled from "styled-components"

const Spacer = styled.div<{ $size?: number }>`
  height: ${({ $size }) => ($size ? `${$size}rem` : "1rem")};
`

export default Spacer
