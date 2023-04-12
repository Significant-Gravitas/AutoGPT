import styled from "styled-components"

const LineSeparatorWithTitle = styled.div<{ title: string }>`
  width: 100%;
  height: 1px;
  background-color: var(--grey100);
  margin-bottom: 1rem;
  margin-top: 1rem;
  // text in middle
  &::before {
    content: "${({ title }) => title?.toUpperCase()}";
    position: relative;
    top: -0.5rem;
    right: -50%;
    display: inline-block;
    font-weight: 500;
    color: var(--primary300);
    padding: 0 0.5rem;
    background-color: var(--grey800);
    transform: translateX(-50%);
  }
`
export default LineSeparatorWithTitle
