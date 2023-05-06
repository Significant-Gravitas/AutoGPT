# PTE Reorder Paragraphs

**Status**: To be implemented

## Input

You are given a set of files split into five groups. Each group corresponds to a paragraph. Each file holds a random sentence from that paragraph.

For each group, choose the most logical order of sentences.
Write the complete paragraph to a file. Example: 1{A-F}.txt -> 1.txt
Respond with an ordered list showing the correct sentence order.

Example:
1. BDAC
2. DCBA
3. CADB

### Files

1A.txt
```
Van Gogh, particularly in his later paintings, creates a thick swirl of paint which perhaps mirrors the emotional storm raging within.
```
1B.txt
```
DuFevre piles the paint onto the canvas in thick swatches which rise off the canvas by a good half inch at times.
```
1C.txt
```
Perhaps the most telling similarity between Van Gogh and DuFevre, the little-known modern surrealist, lies in their use of brushstrokes.
```
1D.txt
```
It is almost as if he is challenging Van Gogh to a contest to determine who was more emotionally disturbed.
```
2A.txt
```
Each week, a new “teleplay” was created from scratch-written, cast, rehearsed and performed.
```
2B.txt
```
Playhouse 90 was truly a remarkable training ground for the young talents.
```
2C.txt
```
Such future luminaries as Paddy Chayefsky, Marlon Brando, and Patricia Neal worked long hours honing their craft.
```
2D.txt
```
In some cases, when there were problems with the censors, it would have to be created twice.
```
3A.txt
```
This is prime tiger country.
```
3B.txt
```
Late March in Ranthambore, the world is all sunshine, crisp air and flowering tress.
```
3C.txt
```
But in the late 1990s, tigers vanished from this open, rugged scrubland along the Aravallis.
```
3D.txt
```
Today, in an ambitious conservation step, work is on here at a frenetic pace to bring back the Lord of the Jungle.
```
4A.txt
```
The 1971 War changed the political geography of the subcontinent.
```
4B.txt
```
Despite the significance of the event, there have been no serious books written about the conflict.
```
4C.txt
```
'Surrender at Dacca' aims to fill the gap.
```
4D.txt
```
It also profoundly alerted the geo-strategic equations in South-East Asia.
```
5A.txt
```
Normally, falling oil prices would boost global growth. This time, though, matters are less clear cut.
```
5B.txt
```
If the source of weakness is financial (debt overhangs and so on), then cheaper oil may not boost growth all that much: consumers may simply use the gains to pay down their debts.
```
5C.txt
```
On the other hand, if plentiful supply is driving prices down, that is potentially better news: cheaper oil should eventually boost spending in the world’s biggest economies.
```
5D.txt
```
The big economic question is whether lower prices reflect weak demand or have been caused by a surge in the supply of crude.
```
5E.txt
```
If weak demand is the culprit, that is worrying: it suggests the oil price is a symptom of weakening growth.
```
5F.txt
```
Indeed, in some countries, cheaper oil may even make matters worse by increasing the risk of deflation.
```

## Output

1. CABD
2. BCAD
3. BACD
4. ADBC
5. ADEBFC

### Files

1.txt
```
Perhaps the most telling similarity between Van Gogh and DuFevre, the little-known modern surrealist, lies in their use of brushstrokes.
Van Gogh, particularly in his later paintings, creates a thick swirl of paint which perhaps mirrors the emotional storm raging within.
DuFevre piles the paint onto the canvas in thick swatches which rise off the canvas by a good half inch at times.
It is almost as if he is challenging Van Gogh to a contest to determine who was more emotionally disturbed.
```

2.txt
```
Playhouse 90 was truly a remarkable training ground for the young talents.
Such future luminaries as Paddy Chayefsky, Marlon Brando, and Patricia Neal worked long hours honing their craft.
Each week, a new “teleplay” was created from scratch-written, cast, rehearsed and performed.
In some cases, when there were problems with the censors, it would have to be created twice.
```

3.txt
```
Late March in Ranthambore, the world is all sunshine, crisp air and flowering tress.
This is prime tiger country.
But in the late 1990s, tigers vanished from this open, rugged scrubland along the Aravallis.
Today, in an ambitious conservation step, work is on here at a frenetic pace to bring back the Lord of the Jungle.
```

4.txt
```
The 1971 War changed the political geography of the subcontinent.
It also profoundly alerted the geo-strategic equations in South-East Asia.
Despite the significance of the event, there have been no serious books written about the conflict.
'Surrender at Dacca' aims to fill the gap.
```

5.txt
```
Normally, falling oil prices would boost global growth. This time, though, matters are less clear cut.
The big economic question is whether lower prices reflect weak demand or have been caused by a surge in the supply of crude.
If weak demand is the culprit, that is worrying: it suggests the oil price is a symptom of weakening growth.
If the source of weakness is financial (debt overhangs and so on), then cheaper oil may not boost growth all that much: consumers may simply use the gains to pay down their debts.
Indeed, in some countries, cheaper oil may even make matters worse by increasing the risk of deflation.
On the other hand, if plentiful supply is driving prices down, that is potentially better news: cheaper oil should eventually boost spending in the world’s biggest economies.
```
