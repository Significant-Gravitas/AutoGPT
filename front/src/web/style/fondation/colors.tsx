import { css } from "styled-components"

const colors = css`
/**
    ██████ ██    ██  █████  ███    ██ 
    ██       ██  ██  ██   ██ ████   ██ 
    ██        ████   ███████ ██ ██  ██ 
    ██         ██    ██   ██ ██  ██ ██ 
    ██████    ██    ██   ██ ██   ████ 
  */

  // cyan : hsl(190, 72%, 48%)
  --cyan-hue: 190;
  --cyan-saturation: 72%;

  --cyan50-lightness: 93%;
  --cyan100-lightness: 88%;
  --cyan200-lightness: 78%;
  --cyan300-lightness: 68%;
  --cyan400-lightness: 58%;
  --cyan500-lightness: 48%;
  --cyan600-lightness: 38%;
  --cyan700-lightness: 28%;
  --cyan800-lightness: 18%;
  --cyan900-lightness: 8%;

  --cyan-lightness: var(--cyan500-lightness);

  // --cyan: hsl(190, 72%, 48%);
  --cyan: hsl(var(--cyan-hue), var(--cyan-saturation), var(--cyan-lightness));
  --cyan-text-color: white;

  // hsl(190, 72%, 88%)
  --cyan100: hsl(
    var(--cyan-hue),
    var(--cyan-saturation),
    var(--cyan100-lightness)
  );

  // hsl(190, 72%, 78%)
  --cyan200: hsl(
    var(--cyan-hue),
    var(--cyan-saturation),
    var(--cyan200-lightness)
  );

  // hsl(190, 72%, 68%)
  --cyan300: hsl(
    var(--cyan-hue),
    var(--cyan-saturation),
    var(--cyan300-lightness)
  );

  // hsl(190, 72%, 58%)
  --cyan400: hsl(
    var(--cyan-hue),
    var(--cyan-saturation),
    var(--cyan400-lightness)
  );

  // hsl(190, 72%, 48%)
  --cyan500: hsl(
    var(--cyan-hue),
    var(--cyan-saturation),
    var(--cyan500-lightness)
  );

  // hsl(190, 72%, 38%)
  --cyan600: hsl(
    var(--cyan-hue),
    var(--cyan-saturation),
    var(--cyan600-lightness)
  );

  // hsl(190, 72%, 28%)
  --cyan700: hsl(
    var(--cyan-hue),
    var(--cyan-saturation),
    var(--cyan700-lightness)
  );

  // hsl(190, 72%, 18%)
  --cyan800: hsl(
    var(--cyan-hue),
    var(--cyan-saturation),
    var(--cyan800-lightness)
  );

  // hsl(190, 72%, 8%)
  --cyan900: hsl(
    var(--cyan-hue),
    var(--cyan-saturation),
    var(--cyan900-lightness)
  );

  /**
    ██████  ██      ██    ██ ███████ 
    ██   ██ ██      ██    ██ ██      
    ██████  ██      ██    ██ █████   
    ██   ██ ██      ██    ██ ██      
    ██████  ███████  ██████  ███████ 
  */

  // blue : hsl(203, 75%, 48%)
  --blue-hue: 203;
  --blue-saturation: 75%;

  --blue50-lightness: 93%;
  --blue100-lightness: 88%;
  --blue200-lightness: 78%;
  --blue300-lightness: 68%;
  --blue400-lightness: 58%;
  --blue500-lightness: 48%;
  --blue600-lightness: 38%;
  --blue700-lightness: 28%;
  --blue800-lightness: 18%;
  --blue900-lightness: 8%;

  --blue-lightness: var(--blue500-lightness);

  // --blue: hsl(203, 75%, 48%);
  --blue: hsl(var(--blue-hue), var(--blue-saturation), var(--blue-lightness));
  --blue-text-color: white;
  // hsl(203, 75%, 88%)
  --blue100: hsl(
    var(--blue-hue),
    var(--blue-saturation),
    var(--blue100-lightness)
  );

  // hsl(203, 75%, 78%)
  --blue200: hsl(
    var(--blue-hue),
    var(--blue-saturation),
    var(--blue200-lightness)
  );

  // hsl(203, 75%, 68%)
  --blue300: hsl(
    var(--blue-hue),
    var(--blue-saturation),
    var(--blue300-lightness)
  );

  // hsl(203, 75%, 58%)
  --blue400: hsl(
    var(--blue-hue),
    var(--blue-saturation),
    var(--blue400-lightness)
  );

  // hsl(203, 75%, 48%)
  --blue500: hsl(
    var(--blue-hue),
    var(--blue-saturation),
    var(--blue500-lightness)
  );

  // hsl(203, 75%, 38%)
  --blue600: hsl(
    var(--blue-hue),
    var(--blue-saturation),
    var(--blue600-lightness)
  );

  // hsl(203, 75%, 28%)
  --blue700: hsl(
    var(--blue-hue),
    var(--blue-saturation),
    var(--blue700-lightness)
  );

  // hsl(203, 75%, 18%)
  --blue800: hsl(
    var(--blue-hue),
    var(--blue-saturation),
    var(--blue800-lightness)
  );

  // hsl(203, 75%, 8%)
  --blue900: hsl(
    var(--blue-hue),
    var(--blue-saturation),
    var(--blue900-lightness)
  );

  /**
     ██████  █████  ██████   █████  ███    ███ ███████ ██      
    ██      ██   ██ ██   ██ ██   ██ ████  ████ ██      ██      
    ██      ███████ ██████  ███████ ██ ████ ██ █████   ██      
    ██      ██   ██ ██   ██ ██   ██ ██  ██  ██ ██      ██      
     ██████ ██   ██ ██   ██ ██   ██ ██      ██ ███████ ███████ 
  */
  --caramel-hue: 27;
  --caramel-saturation: 74%;

  --caramel50-lightness: 90%;
  --caramel100-lightness: 85%;
  --caramel200-lightness: 75%;
  --caramel300-lightness: 65%;
  --caramel400-lightness: 55%;
  --caramel500-lightness: 45%;
  --caramel600-lightness: 35%;
  --caramel700-lightness: 25%;
  --caramel800-lightness: 15%;
  --caramel900-lightness: 5%;

  --caramel-lightness: var(--caramel400-lightness);

  // --caramel: hsl(27, 74%, 55%);
  --caramel: hsl(
    var(--caramel-hue),
    var(--caramel-saturation),
    var(--caramel-lightness)
  );
  --caramel-text-color: white;

  // hsl(27, 74%, 85%)
  --caramel100: hsl(
    var(--caramel-hue),
    var(--caramel-saturation),
    var(--caramel100-lightness)
  );

  // hsl(27, 74%, 75%)
  --caramel200: hsl(
    var(--caramel-hue),
    var(--caramel-saturation),
    var(--caramel200-lightness)
  );

  // hsl(27, 74%, 65%)
  --caramel300: hsl(
    var(--caramel-hue),
    var(--caramel-saturation),
    var(--caramel300-lightness)
  );

  // hsl(27, 74%, 55%)
  --caramel400: hsl(
    var(--caramel-hue),
    var(--caramel-saturation),
    var(--caramel400-lightness)
  );

  // hsl(27, 74%, 45%)
  --caramel500: hsl(
    var(--caramel-hue),
    var(--caramel-saturation),
    var(--caramel500-lightness)
  );

  // hsl(27, 74%, 35%)
  --caramel600: hsl(
    var(--caramel-hue),
    var(--caramel-saturation),
    var(--caramel600-lightness)
  );

  // hsl(27, 74%, 25%)
  --caramel700: hsl(
    var(--caramel-hue),
    var(--caramel-saturation),
    var(--caramel700-lightness)
  );

  // hsl(27, 74%, 15%)
  --caramel800: hsl(
    var(--caramel-hue),
    var(--caramel-saturation),
    var(--caramel800-lightness)
  );

  // hsl(27, 74%, 5%)
  --caramel900: hsl(
    var(--caramel-hue),
    var(--caramel-saturation),
    var(--caramel900-lightness)
  );

    /**
    ██    ██ ███████ ██      ██       ██████  ██     ██ 
    ██  ██  ██      ██      ██      ██    ██ ██     ██ 
    ████   █████   ██      ██      ██    ██ ██  █  ██ 
     ██    ██      ██      ██      ██    ██ ██ ███ ██ 
     ██    ███████ ███████ ███████  ██████   ███ ███  
  */
  --yellow-hue: 67;
  --yellow-saturation: 93%;

  --yellow50-lightness: 89%;
  --yellow100-lightness: 83%;
  --yellow-200-lightness: 73%;
  --yellow300-lightness: 63%;
  --yellow400-lightness: 53%;
  --yellow500-lightness: 43%;
  --yellow600-lightness: 33%;
  --yellow700-lightness: 23%;
  --yellow800-lightness: 13%;
  --yellow900-lightness: 3%;

  --yellow-lightness: var(--yellow300-lightness);

  // --yellow: hsl(67, 93%, 63%)
  --yellow: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow-lightness)
  );
  --yellow-text-color: black;

  // hsl(67, 93%, 89%)
  --yellow50: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow50-lightness)
  );

  // hsl(67, 93%, 83%)
  --yellow100: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow100-lightness)
  );

  // hsl(67, 93%, 73%)
  --yellow200: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow200-lightness)
  );

  // hsl(67, 93%, 63%)
  --yellow300: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow300-lightness)
  );

  // hsl(67, 93%, 53%);
  --yellow400: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow400-lightness)
  );

  // hsl(67, 93%, 43%)
  --yellow500: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow500-lightness)
  );

  // hsl(67, 93%, 33%)
  --yellow600: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow600-lightness)
  );

  // hsl(67, 93%, 23%)
  --yellow700: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow700-lightness)
  );

  // hsl(67, 93%, 13%)
  --yellow800: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow800-lightness)
  );

  // hsl(67, 93%, 3%)
  --yellow900: hsl(
    var(--yellow-hue),
    var(--yellow-saturation),
    var(--yellow900-lightness)
  );


  /**
    ███████ ███    ███ ███████ ██████   █████  ██      ██████  
    ██      ████  ████ ██      ██   ██ ██   ██ ██      ██   ██ 
    █████   ██ ████ ██ █████   ██████  ███████ ██      ██   ██ 
    ██      ██  ██  ██ ██      ██   ██ ██   ██ ██      ██   ██ 
    ███████ ██      ██ ███████ ██   ██ ██   ██ ███████ ██████  
  */

  --emerald-hue: 149;
  --emerald-saturation: 69%;

  --emerald50-lightness: 96%;
  --emerald100-lightness: 91%;
  --emerald200-lightness: 81%;
  --emerald300-lightness: 71%;
  --emerald400-lightness: 61%;
  --emerald500-lightness: 51%;
  --emerald600-lightness: 41%;
  --emerald700-lightness: 31%;
  --emerald800-lightness: 21%;
  --emerald900-lightness: 11%;

  --emerald-lightness: var(--emerald500-lightness);
  --emerald-text-color: white;
  // --emerald: hsl(149, 69%, 51%);
  --emerald: hsl(
    var(--emerald-hue),
    var(--emerald-saturation),
    var(--emerald-lightness)
  );

  // hsl(149, 69%, 96%)
  --emerald50: hsl(
    var(--emerald-hue),
    var(--emerald-saturation),
    var(--emerald50-lightness)
  );

  // hsl(149, 69%, 91%)
  --emerald100: hsl(
    var(--emerald-hue),
    var(--emerald-saturation),
    var(--emerald100-lightness)
  );

  // hsl(149, 69%, 81%)
  --emerald200: hsl(
    var(--emerald-hue),
    var(--emerald-saturation),
    var(--emerald200-lightness)
  );

  // hsl(149, 69%, 71%)
  --emerald300: hsl(
    var(--emerald-hue),
    var(--emerald-saturation),
    var(--emerald300-lightness)
  );

  // hsl(149, 69%, 61%)
  --emerald400: hsl(
    var(--emerald-hue),
    var(--emerald-saturation),
    var(--emerald400-lightness)
  );

  // hsl(149, 69%, 51%)
  --emerald500: hsl(
    var(--emerald-hue),
    var(--emerald-saturation),
    var(--emerald500-lightness)
  );

  // hsl(149, 69%, 41%)
  --emerald600: hsl(
    var(--emerald-hue),
    var(--emerald-saturation),
    var(--emerald600-lightness)
  );

  // hsl(149, 69%, 31%)
  --emerald700: hsl(
    var(--emerald-hue),
    var(--emerald-saturation),
    var(--emerald700-lightness)
  );

  // hsl(149, 69%, 21%)
  --emerald800: hsl(
    var(--emerald-hue),
    var(--emerald-saturation),
    var(--emerald800-lightness)
  );

  // hsl(149, 69%, 11%)
  --emerald900: hsl(
    var(--emerald-hue),
    var(--emerald-saturation),
    var(--emerald900-lightness)
  );

  //   ██████  ██    ██ ██████  ██████  ██      ███████ 
  // ██   ██ ██    ██ ██   ██ ██   ██ ██      ██      
  // ██████  ██    ██ ██████  ██████  ██      █████   
  // ██      ██    ██ ██   ██ ██      ██      ██      
  // ██       ██████  ██   ██ ██      ███████ ███████ 
                                                  
                                                  

  // purple : hsl(244, 50%, 65%)
  --purple-hue: 244;
  --purple-saturation: 50%;

  --purple100-lightness: 95%;
  --purple200-lightness: 85%;
  --purple300-lightness: 75%;
  --purple400-lightness: 65%;
  --purple500-lightness: 55%;
  --purple600-lightness: 45%;
  --purple700-lightness: 35%;
  --purple800-lightness: 25%;
  --purple900-lightness: 15%;

  --purple-lightness: var(--purple400-lightness);

  // hsl(244, 50%, 95%)
  --purple100: hsl(
    var(--purple-hue),
    var(--purple-saturation),
    var(--purple100-lightness)
  );

  // hsl(244, 50%, 85%)
  --purple200: hsl(
    var(--purple-hue),
    var(--purple-saturation),
    var(--purple200-lightness)
  );

  // hsl(244, 50%, 75%)
  --purple300: hsl(
    var(--purple-hue),
    var(--purple-saturation),
    var(--purple300-lightness)
  );

  // hsl(244, 50%, 65%)
  --purple400: hsl(
    var(--purple-hue),
    var(--purple-saturation),
    var(--purple400-lightness)
  );

  // hsl(244, 50%, 55%)
  --purple500: hsl(
    var(--purple-hue),
    var(--purple-saturation),
    var(--purple500-lightness)
  );

  // hsl(244, 50%, 45%)
  --purple600: hsl(
    var(--purple-hue),
    var(--purple-saturation),
    var(--purple600-lightness)
  );

  // hsl(244, 50%, 35%)
  --purple700: hsl(
    var(--purple-hue),
    var(--purple-saturation),
    var(--purple700-lightness)
  );

  // hsl(244, 50%, 25%)
  --purple800: hsl(
    var(--purple-hue),
    var(--purple-saturation),
    var(--purple800-lightness)
  );

  // hsl(244, 50%, 15%)
  --purple900: hsl(
    var(--purple-hue),
    var(--purple-saturation),
    var(--purple900-lightness)
  );

  --purple: hsl(
    var(--purple-hue),
    var(--purple-saturation),
    var(--purple-lightness)
  );
  --purple-text-color: white;

  //    ██████  ██████  ███████ ██    ██ 
  // ██       ██   ██ ██       ██  ██  
  // ██   ███ ██████  █████     ████   
  // ██    ██ ██   ██ ██         ██    
  //  ██████  ██   ██ ███████    ██    
                                    
                                  

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

  //   ██████  ██████  ██ ███    ███  █████  ██████  ██    ██ 
  // ██   ██ ██   ██ ██ ████  ████ ██   ██ ██   ██  ██  ██  
  // ██████  ██████  ██ ██ ████ ██ ███████ ██████    ████   
  // ██      ██   ██ ██ ██  ██  ██ ██   ██ ██   ██    ██    
  // ██      ██   ██ ██ ██      ██ ██   ██ ██   ██    ██    

  --primary: var(--yellow);
  --primary-hue: var(--yellow-hue);
  --primary-saturation: var(--yellow-saturation);
  --primary-lightness: var(--yellow-lightness);
  --primary-text-color: var(--yellow-text-color);
  --primary50: var(--yellow50);
  --primary50-lightness: var(--yellow50-lightness);
  --primary100: var(--yellow100);
  --primary100-lightness: var(--yellow100-lightness);
  --primary200: var(--yellow200);
  --primary200-lightness: var(--yellow200-lightness);
  --primary300: var(--yellow300);
  --primary300-lightness: var(--yellow300-lightness);
  --primary400: var(--yellow400);
  --primary400-lightness: var(--yellow400-lightness);
  --primary500: var(--yellow500);
  --primary500-lightness: var(--yellow500-lightness);
  --primary600: var(--yellow600);
  --primary600-lightness: var(--yellow600-lightness);
  --primary700: var(--yellow700);
  --primary700-lightness: var(--yellow700-lightness);
  --primary800: var(--yellow800);
  --primary800-lightness: var(--yellow800-lightness);
  --primary900: var(--yellow900);
  --primary900-lightness: var(--yellow900-lightness);
`

export default colors
