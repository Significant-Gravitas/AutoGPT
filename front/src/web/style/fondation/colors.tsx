import { css } from "styled-components"

const colors = css`
  --yellow-hue: 50;
  --yellow-saturation: 100%;

  --yellow50-lightness: 93%;
  --yellow100-lightness: 88%;
  --yellow-200-lightness: 78%;
  --yellow300-lightness: 68%;
  --yellow400-lightness: 58%;
  --yellow500-lightness: 48%;
  --yellow600-lightness: 38%;
  --yellow700-lightness: 28%;
  --yellow800-lightness: 18%;
  --yellow900-lightness: 8%;

  --yellow-lightness: var(--yellow300-lightness);

  // --yellow: hsl(50, 100%, 68%);
  --yellow: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow-lightness)
  );
  --yellow-text-color: black;

  // hsl(50, 100%, 88%)
  --yellow100: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow100-lightness)
  );

  // hsl(50, 100%, 78%)
  --yellow200: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow200-lightness)
  );

  // hsl(50, 100%, 68%)
  --yellow300: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow300-lightness)
  );

  // hsl(50, 100%, 58%)
  --yellow400: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow400-lightness)
  );

  // hsl(50, 100%, 48%)
  --yellow500: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow500-lightness)
  );

  // hsl(50, 100%, 38%)
  --yellow600: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow600-lightness)
  );

  // hsl(50, 100%, 28%)
  --yellow700: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow700-lightness)
  );

  // hsl(50, 100%, 18%)
  --yellow800: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow800-lightness)
  );

  // hsl(50, 100%, 8%)
  --yellow900: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow900-lightness)
  );

  // grey : hsl(180, 1%, 36%);
  --grey-hue: 180;
  --grey-saturation: 1%;

  --grey-lightness: 36%;

  --grey50-lightness: 96%;
  --grey100-lightness: 86%;
  --grey200-lightness: 76%;
  --grey300-lightness: 66%;
  --grey400-lightness: 56%;
  --grey500-lightness: 46%;
  --grey600-lightness: 36%;
  --grey700-lightness: 26%;
  --grey800-lightness: 16%;
  --grey900-lightness: 6%;

  --grey-lightness: var(--grey600-lightness);

  // --grey: hsl(180, 1%, 36%);
  --grey: hsl(var(--grey-hue), var(--grey-saturation), var(--grey-lightness));
  --grey-text-color: white;

  // hsl(162, 9%, 93%)
  --grey50: hsl(
    var(--grey-hue),
    var(--grey-saturation),
    var(--grey50-lightness)
  );

  // hsl(162, 9%, 88%)
  --grey100: hsl(
    var(--grey-hue),
    var(--grey-saturation),
    var(--grey100-lightness)
  );

  // hsl(162, 9%, 78%)
  --grey200: hsl(
    var(--grey-hue),
    var(--grey-saturation),
    var(--grey200-lightness)
  );

  // hsl(162, 9%, 68%)
  --grey300: hsl(
    var(--grey-hue),
    var(--grey-saturation),
    var(--grey300-lightness)
  );

  // hsl(162, 9%, 58%)
  --grey400: hsl(
    var(--grey-hue),
    var(--grey-saturation),
    var(--grey400-lightness)
  );

  // hsl(162, 9%, 48%)
  --grey500: hsl(
    var(--grey-hue),
    var(--grey-saturation),
    var(--grey500-lightness)
  );

  // hsl(162, 9%, 38%)
  --grey600: hsl(
    var(--grey-hue),
    var(--grey-saturation),
    var(--grey600-lightness)
  );

  // hsl(162, 9%, 28%)
  --grey700: hsl(
    var(--grey-hue),
    var(--grey-saturation),
    var(--grey700-lightness)
  );

  // hsl(162, 9%, 18%)
  --grey800: hsl(
    var(--grey-hue),
    var(--grey-saturation),
    var(--grey800-lightness)
  );

  // hsl(162, 9%, 8%)
  --grey900: hsl(
    var(--grey-hue),
    var(--grey-saturation),
    var(--grey900-lightness)
  );
`

export default colors
