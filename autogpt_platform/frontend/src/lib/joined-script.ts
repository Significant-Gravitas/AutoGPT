const JOINED =
  /[\p{Script=Arabic}\p{Script=Syriac}\p{Script=Nko}\p{Script=Mongolian}\p{Script=Adlam}\p{Script=Hanifi_Rohingya}\p{Script=Mandaic}]/u;

export const joinsLetters = (text: string) => JOINED.test(text);
