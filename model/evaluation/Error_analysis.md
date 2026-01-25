## Results Eval round 1 (eval_result_2)

(for # Questions which do not have expected_answer, gen metrics are not calculated automatically)
(Multi-turn questions are evaluated in eval_result_3)

# # Question: 1 (ingredients)
Query rewriting: correct
Generated answer: correct 
Top 1 doc: correct 
Chunk #1: correct 
Section #1: correct
EM accuracy: 0.0
Manual accuracy: 1.0
F1: 0.0
Cosine-similarity: 0.7948027849197388
Num chunks retrieved: 
Notes: ingredient measurements included → lowering EM accuracy 
**Error type**: None 


# # Question: 2 (ingredients)
Query rewriting: correct
Generated answer: correct 
Doc #1: correct 
Chunk #1: correct 
Section #1: correct
EM accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 0.9549620747566223
Num chunks retrieved: 9
Notes: all perfect
**Error type**: None 

# Question: 3 (ingredients)
Query rewriting: correct
Generated answer: hallucination - added (sugar, egg yolks, sugar,), omission: cocoa powder
Doc #1: correct 
Chunk #1: correct 
Section #1: correct
EM accuracy: 0.0
Manual accuracy: 0.75
F1: 0.0
Cosine-similarity: 0.741157054901123
Num chunks retrieved: 5
Notes: hallucination and omission of ingredients
**Error type**: Generation- repetition + omission  

# Question: 4 (ingredients, binary)  
Query rewriting: correct
Generated answer: correct 
Doc #1: correct
Chunk #1: correct 
Section #1: correct 
EM accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 2
Notes: correct
**Error type**: None 

# Question: 5 (ingredients)
Query rewriting: correct
Generated answer: correct
Doc #1: correct
Chunk #1: correct 
Section #1: correct 
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 2
Notes: -
**Error type**: None 

# Question: 6 (ingredients) 
Query rewriting: correct
Generated answer: correct 
Doc #1: correct
Chunk #1: correct
Section #1: correct
EM Accuracy: 0.0
Manual accuracy: 1.0
F1: 0.0
Cosine-similarity: 0.8109897375106812
Num chunks retrieved: 5
Notes: ingredient measurement added
**Error type**: None 

###### Ignored 
# Question: 7 (ingredients) 
Query rewriting: incorrect → some sort of ‘leakage’ from previous # Question → Q ignored
Generated answer: 
Doc #1: 
Chunk #1:
Section #1:
EM Accuracy: 
Manual accuracy: 
F1: 
Cosine-similarity: 
Num chunks retrieved: 
Notes: # Question ignored due to query rewrite ‘leakage’ 

# Question: 8 (ingredients)
Query rewriting: correct
Generated answer: correct 
Doc #1: correct
Chunk #1: incorrect (chunk 1: info, chunk 2: steps 0 (partially correct → info found there as well), 3: steps 5, 4: steps 4, 5: ingredients (correct one) 
Section #1: incorrect
EM Accuracy: 0.666
Manual accuracy: 1.0
F1: 0.666
Cosine-similarity: 0.5978608131408691
Num chunks retrieved: 10
Notes: golden chunks not top 3
**Error type**: None 

# Question: 9 (ingredients) (binary) 
Query rewriting: correct 
Generated answer: correct
Doc #1: correct
Chunk #1: incorrect (steps 0, but answer also found there), (correct chunk 2)
Section #1: incorrect (steps)
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 3
Notes: not top chunk retrieved but in top 2
**Error type**: None 

# Question: 10 (instructions)
Query rewriting: correct
Generated answer: incorrect
Doc #1: correct
Chunk #1 incorrect (1: ingredients)
Section #1: incorrect (ingredients) 
EM Accuracy: -
Manual accuracy: 0.0
F1: 0.0
Cosine-similarity: -
Num chunks retrieved: 1
Notes: incorrect tag given to # Question (ingredients), listed ingredients and not steps
**Error type**: Wrong filter → Generation error: # Question-answer misalignment (QAM) 

# Question:  11 (instructions)
Query rewriting:  correct 
Generated answer: partial, lists the first step (1/5)
Doc #1: correct
Chunk #5: (1: steps 0, 2: steps 4, 3: steps 5, 4: ingredients, 5: info, 6: steps (wrong recipe), 7: steps 6 (wrong recipe), 8: steps 0 (wrong recipe), 9: steps (wrong recipe)
Section #1: correct) → 3/5
EM Accuracy: -
Manual accuracy: 0.2	
F1: -
Cosine-similarity: -
Num chunks retrieved: 9
Notes: omission of 4/5 steps of recipe 
**Error type**: Generation: omission 

# Question: 12 (instructions)
Query rewriting: correct
Generated answer: partial, 1/4 
Doc #1: correct 
Chunk #4: ¾ (chunks 6-8 retrieved wrong recipe) 
Section #1: correct 
EM Accuracy: -
Manual accuracy: 0.25
F1: -
Cosine-similarity: -
Num chunks retrieved: 8
Notes: partial generation, omission of steps
**Error type**: Generation: omission 

# Question: 13 (instructions)
Query rewriting: correct
Generated answer: 1/12 steps partial generation 
Doc #1: correct
Chunk #12: (1: steps 0, 2: 12, 3: 1,  4: wrong recipe, 5: info, 6: 13, 7: wrong recipe, 8: ingredients, 9: wrong recipe) 3/12
Section #1: correct 
EM Accuracy: -
Manual accuracy: 0.083
F1: -
Cosine-similarity: -
Num chunks retrieved: 9
Notes: omission of many steps 
**Error type**: Omission 

# Question: 14 (instructions/general)
Query rewriting: correct 
Generated answer: incorrect 
Doc #1: correct
Chunk #2: (1: step 7 (correct, 2: info (partially correct), 3:0, 4: wrong recipe, 5: wrong recipe, 6: wrong recipe (WR), 7: 8, 8: WR, 9: ingredients, 10: WR) 1/2 
Section #1: correct 
EM Accuracy: -
Manual accuracy: 0.0
F1: -
Cosine-similarity: -
Num chunks retrieved: 10
Notes: many chunks retrieved, answer incorrect
**Error type**: partial Chunk retrieval, Generation: grounded incorrect answer (context dilution) 

# Question: 15 (instructions) 
Query rewriting: correct
Generated answer: correct 
Doc #1: correct (but correct chunk in top 2) (chunks 6-8 WR)
Chunk #1: incorrect
Section #1: correct 
EM Accuracy: -
Manual accuracy: 1.0
F1: -
Cosine-similarity: -
Num chunks retrieved: 
Notes: correct
**Error type**: None 

# Question: 16 (instructions)
Query rewriting: correct 
Generated answer: incorrect
Doc #1: incorrect (from all chunks, correct recipe was never retrieved)
Chunk #1: incorrect 
Section #1: correct 
EM Accuracy: -
Manual accuracy: 0.0
F1: -
Cosine-similarity: -
Num chunks retrieved: 4
Notes: wrong recipe retrieved (in instructions it mentions christmas holidays) and also does not explain how to prepare them 
**Error type**: Retrieval error: document level, 

# Question: 17 (general)
Query rewriting: correct 
Generated answer: incorrect, does not mention name of the dish 
Doc #1: correct (christmas dish)
Chunk #1: -
Section #1:-
EM Accuracy: -
Manual accuracy: 0.0
F1: 
Cosine-similarity: 
Num chunks retrieved: 1
Notes: does not mention christmas dish it suggested
**Error type**: Generation: omission required details 

###### Ignored 
# Question: 18 (general) 
Query rewriting: incorrect, for some reason used christmas canapes in query rewriting, ignore results
Generated answer: 
Doc #1: 
Chunk #1:
Section #1:
EM Accuracy: 
Manual accuracy: 
F1: 
Cosine-similarity: 
Num chunks retrieved: 
Notes: did correctly tag it with seafood 

###### Ignored 
# Question: 19 (general) 
Query rewriting: rewrote incorrectly again ignore results
Generated answer: 
Doc #1: 
Chunk #1:
Section #1:
EM Accuracy: 
Manual accuracy: 
F1: 
Cosine-similarity: 
Num chunks retrieved: 
Notes: 

# Question: 20 (general)
Query rewriting: correct
Generated answer: correct 
Doc #1: correct
Chunk #1: partially correct (step 14 talks about reheating in oven,, 2: 13 (correct even though not in gold id) (6-9 wr) 
Section #1: correct 
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 9
Notes: correct
**Error type**: None (chunk retrieval) 

###### Ignored 
# Question: 21 (general) 
Query rewriting: incorrect (used lasagna pockets), ignore 
Generated answer: 
Doc #1: 
Chunk #1:
Section #1:
EM Accuracy: 
Manual accuracy: 
F1: 
Cosine-similarity: 
Num chunks retrieved: 
Notes: 

# Question: 22 (ingredients)
Query rewriting: correct 
Generated answer: incorrect (wrong amount, but correct format)
Doc #1: correct
Chunk #1: incorrect
Section #1: incorrect
EM Accuracy: 0.0
Manual accuracy: 0.75
F1: 0.0
Cosine-similarity: 0.6459773778915405
Num chunks retrieved: 10
Notes: never retrieved correct chunk, therefore answer is correct format but wrong answer
**Error type**: Retrieval: chunk retrieval 

# Question: 23 (general)
Query rewriting: correct
Generated answer: incorrect (correct format, incorrect answer)
Doc #1: incorrect (never retrieved correct document)
Chunk #1: incorrect 
Section #1: correct
EM Accuracy: 0.0
Manual accuracy: 0.2
F1: 0.0
Cosine-similarity: 0.7133795619010925
Num chunks retrieved: 3
Notes: never retrieved correct recipe, so answer in correct format but answer wrong 
**Error type**: Retrieval: document retrieval 

# Question: 24.1 (multi-turn) (ingredients)
Query rewriting: correct 
Generated answer: incorrect 
Doc #1: incorrect 
Chunk #1: Incorrect
Section #1: incorrect 
EM Accuracy: 0.0
Manual accuracy: 0.0
F1: 0.0
Cosine-similarity: 0.16049692034721375
Num chunks retrieved: 1
Notes: incorrect on: retrieval + generation 
**Error type**: retrieval: document

# Question: 24.2 (multi-turn) (ingredients) 
Query rewriting: partially correct (mushroom meatloaf → meatloaf)
Generated answer: partially correct 
Doc #1: correct
Chunk #1: incorrect  (correct chunk: #6)
Section #1: correct
EM Accuracy:  0.19999999999999998
Manual accuracy: 
F1: 0.19999999999999998
Cosine-similarity: 0.4729187786579132
Num chunks retrieved: 10
Notes: Partially correct answer, correct doc retrieved but correct chunk not in top-5 k chunks
**Error type**: Generation: incomplete answer

# Question: 25 (general)
Query rewriting: correct 
Generated answer: partially, irrelevant info included + answer repeated
Doc #1: incorrect  (but in top 2) 
Chunk #1: incorrect (but in top 2) 
Section #1: incorrect (but in top 2)
EM Accuracy: 1.0
Manual accuracy: 0.1
F1: 0.038461538461538464
Cosine-similarity: 0.19092094898223877
Num chunks retrieved: 9
Notes: addition of irrelevant info in the answer
**Error type**: generation: addition of irrelevance 

# Question: 26 (general)
Query rewriting: correct
Generated answer: wrong (correct format, but wrong answer) 
Doc #1: incorrect (correct recipe not in top k chunks) 
Chunk #1: incorrect 
Section #1: incorrect 
EM Accuracy: 0.0
Manual accuracy: 0.0
F1: 0.0
Cosine-similarity: 0.22782358527183533
Num chunks retrieved: 2
Notes: answer format is correct, but incorrect documents retrieved → incorrect generation 
**Error type**: Retrieval: document retrieval 

# Question: 27 (general)
Query rewriting: correct
Generated answer: incorrect (correct format, but wrong answer) 
Doc #1: incorrect 
Chunk #1: incorrect 
Section #1: correct 
EM Accuracy: 0.25
Manual accuracy: 0.15
F1: 0.25
Cosine-similarity: 0.3990623950958252
Num chunks retrieved: 1
Notes:  correct format, but incorrect document retrieved, so answer is incorrect 
**Error type**: retrieval: document 

# Question: 28 (general)
Query rewriting: correct 
Generated answer: correct 
Doc #1: correct 
Chunk #1: incorrect 
Section #1: incorrect (but in top 2) 
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 3
Notes: correct 
**Error type**: None 

# Question: 29 (general) 
Query rewriting:  correct 
Generated answer: correct (addition of (semi) irrelevant info
Doc #1: correct
Chunk #1: correct
Section #1: correct
EM Accuracy: 0.125
Manual accuracy: 0.333
F1: 0.125
Cosine-similarity: 0.5352104902267456
Num chunks retrieved:10 
Notes: correct, but addition of (semi) irrelevant info 
**Error type**: None 

# Question: 30 (general/ingredients)
Query rewriting: correct 
Generated answer: correct 
Doc #1: correct
Chunk #1: correct 
Section #1: correct 
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 10
Notes:  correct, but retrieved a lot of documents
**Error type**: None 

# Question: 31 (instructions) 
Query rewriting: correct 
Generated answer: correct
Doc #1: correct 
Chunk #1: incorrect (correct in top 2) 
Section #1: incorrect (in top 2) 
EM Accuracy: 0.7368421052631579 → changed to 1.0 (expected_answer not entirely correct) 
Manual accuracy: 1.0
F1: 0.7368421052631579 → 1.0 
Cosine-similarity: 0.9756057262420654
Num chunks retrieved: 9
Notes:  correct, expected answer had additional irrelevant info 
**Error type**: None 

###### Ignored 
# Question: 32.1 (multi-turn)
Query rewriting: incorrect → something went wrong, ignore answers
Generated answer: 
Doc #1: 
Chunk #1: 
Section #1: 
EM Accuracy: 
Manual accuracy: 
F1: 
Cosine-similarity: 
Num chunks retrieved: 
Notes:  

###### Ignored 
# Question: 33 (multi-turn)
Query rewriting: incorrect → something went wrong 
Generated answer: 
Doc #1: 
Chunk #1: 
Section #1: 
EM Accuracy: 
Manual accuracy: 
F1: 
Cosine-similarity: 
Num chunks retrieved: 
Notes:  

# Question: 34 (ingredients)
Query rewriting: correct
Generated answer: correct
Doc #1: correct 
Chunk #1: correct
Section #1: correct
EM Accuracy: 0.4
Manual accuracy: 1.0
F1: 0.4
Cosine-similarity: 0.6070601344108582
Num chunks retrieved: 8
Notes: correct, phrased differently so lower accuracy score
**Error type**: None 

# Question: 35 (ingredients) 
Query rewriting: correct 
Generated answer: incorrect (correct format, incorrect answer) 
Doc #1: correct 
Chunk #1: incorrect  (but in top 3) 
Section #1: incorrect (in top 3) 
EM Accuracy: 0.5
Manual accuracy: 0.5
F1: 0.5
Cosine-similarity: 0.6367925405502319
Num chunks retrieved: 10
Notes:  correct format, but wrong part of recipe used (wrong chunk) → wrong type 
**Error type**: Retrieval: chunk 

# Question: 36 (general)
Query rewriting: correct 
Generated answer: incorrect 
Doc #1: incorrect (correct never retrieved) 
Chunk #1: incorrect 
Section #1: correct
EM Accuracy: 0.0
Manual accuracy: 0.0
F1: 0.0
Cosine-similarity: 0.489060640335083
Num chunks retrieved: 3 
Notes:  format not necessarily wrong, but addition of irrelevant info + wrong recipe retrieved
**Error type**: Retrieval: document, Generation: addition 

# Question: 37 (instructions) 
Query rewriting: correct 
Generated answer: incorrect (wrong step, but correct format) 
Doc #1: correct
Chunk #1: incorrect (correct chunk never retrieved)
Section #1: incorrect
EM Accuracy: 0.30303030303030304
Manual accuracy: 0.0
F1: 0.30303030303030304
Cosine-similarity: 0.7197442054748535
Num chunks retrieved: 6
Notes:  incorrect chunks retrieved → incorrect answer, but correct format
**Error type**: Retrieval: chunk 

# Question: 38 (general)
Query rewriting: correct 
Generated answer: incorrect
Doc #3: correct
Chunk #1: -
Section #1: -
EM Accuracy: -
Manual accuracy: 0.0
F1: -
Cosine-similarity: -
Num chunks retrieved: 10
Notes: does not mention the recipes, output is repeated many times. Correctly retrieves desserts containing fruits
**Error type**: Generation: omission + repetition 

# Question: 39 (general/instructions)
Query rewriting: correct 
Generated answer: correct
Doc #1: correct
Chunk #1: incorrect (but in top 2)
Section #1: correct
EM Accuracy: 0.3076923076923077
Manual accuracy: 1.0
F1:  0.3076923076923077
Cosine-similarity: 0.2839388847351074
Num chunks retrieved: 10
Notes: answer formulated differently to expected answer, causing drop in accuracy 
**Error type**: None 

# Question: 40 (ingredients)
Query rewriting:  correct
Generated answer: correct
Doc #1: incorrect (correct #4)
Chunk #1: incorrect (correct #4)
Section #1: incorrect
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 10
Notes: correct answer, gold chunk/doc in top 5
**Error type**: None 

# Question: 41 (ingredients) 
Query rewriting: correct 
Generated answer: correct 
Doc #1: correct
Chunk #1: correct
Section #1: correct
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 0.8571428571428571
Cosine-similarity: 0.9349097609519958
Num chunks retrieved: 7
Notes:  correct 
**Error type**: None

###### Ignored 
# Question: 42.1 (multi-turn)
Query rewriting: ignore # Question did not work again (maybe found reason why, uses text instead of ‘# Question’, but did you # Question in first multi-turn) 
Generated answer: 
Doc #1: 
Chunk #1: 
Section #1: 
EM Accuracy: 
Manual accuracy: 
F1: 
Cosine-similarity: 
Num chunks retrieved: 
Notes:  

# Question: 43 (general/ingredients) 
Query rewriting: correct 
Generated answer: incorrect (correct format, incorrect answer)
Doc #1: correct 
Chunk #1: incorrect (Correct #8)
Section #1: incorrect 
EM Accuracy: 0.0
Manual accuracy: 0.0
F1: 0.0
Cosine-similarity: 0.7334951162338257
Num chunks retrieved: 10
Notes: correct format answer, but wrong answer
**Error type**: retrieval: document 

# Question: 44 (general/instructions)
Query rewriting: correct
Generated answer: correct 
Doc #1: correct
Chunk #1: correct (not gold, but still good, step9)
Section #1: correct (not gold, but still correct)
EM Accuracy: 0.5142857142857142
Manual accuracy: 1.0
F1: 0.5142857142857142
Cosine-similarity: 0.7467948198318481
Num chunks retrieved: 10
Notes: correct answer, just formulated differently, causing drop in accuracy 
**Error type**: None 

# Question: 45 (general/ingredients)
Query rewriting: correct
Generated answer: incorrect (correct format, incorrect answer) 
Doc #1: correct 
Chunk #1: correct 
Section #1: correct
EM Accuracy: 0.0
Manual accuracy: 0.0
F1: 0.0
Cosine-similarity: 0.7334951162338257
Num chunks retrieved: 10
Notes: correct chunks retrieved, format correct, but answer is wrong
**Error type**: Generation: hallucination 

# Question: 46 (general)
Query rewriting: correct 
Generated answer: correct
Doc #1: correct 
Chunk #1: incorrect (but in top-3)
Section #1: incorrect 
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 9
Notes:  correct answer
**Error type**: None 

###### Ignored 
# Question: 47 (general)
Query rewriting: correct
Generated answer: “the context does not contain the required info”, should have retrieved recipes and be able to answer
Doc #1: 1: correct, 2: incorrect, 3: incorrect, 4: correct, 5: incorrect, 6: incorrect, 7: correct, 
Chunk #1: -
Section #1: -
EM Accuracy: -
Manual accuracy: 0.0
F1: -
Cosine-similarity: -
Num chunks retrieved: 7, 
Notes:  should have been able to answer, 3/7 correct type of recipe (of chunks) 

# Question: 48 (instructions)
Query rewriting: correct
Generated answer: incorrect (correct format, incorrect recipe) 
Doc #1: incorrect 
Chunk #1: incorrect 
Section #1: incorrect 
EM Accuracy: 0.10169491525423728
Manual accuracy: 0.0
F1: 0.10169491525423728
Cosine-similarity: 0.12053632736206055
Num chunks retrieved: 3
Notes:  got incorrect tag of ‘dessert’, so could not retrieve correct type of docs
**Error type**: Retrieval: document 

# Question: 49 (ingredients)
Query rewriting: correct
Generated answer: correct 
Doc #1: correct
Chunk #1: incorrect (#4 so within top-5)
Section #1: incorrect
EM Accuracy: 1.0 
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 0.5867729187011719
Num chunks retrieved: 10
Notes:  correct
**Error type**: None 

# Question: 50 (ingredients)
Query rewriting: correct
Generated answer: correct
Doc #1: correct
Chunk #1: correct
Section #1: correct
EM Accuracy: 0.0
Manual accuracy: 1.0
F1: 0.0
Cosine-similarity: 0.8195382356643677
Num chunks retrieved: 3
Notes:  correct, just phrased differently (added measurements)
**Error type**: None 

# Question: 51 (background) 
Query rewriting: correct
Generated answer: correct 
Doc #1: did not retrieve the wiki doc/chunk, but answer can be found in docs retrieved
Chunk #1: -
Section #1: -
EM Accuracy: 0.18181818181818182
Manual accuracy: 1.0
F1: 0.18181818181818182
Cosine-similarity: 0.6367530226707458
Num chunks retrieved: 10
Notes: correct answer retrieved from other docs than golden (info was inside those chunks) 
**Error type**: Retrieval: failure 

# Question: 52 (background) 
Query rewriting: correct 
Generated answer: ‘the context does not contain the required information’, possibly due to not being able to find the wiki docs 
Doc #1: 
Chunk #1: 
Section #1: 
EM Accuracy: 
Manual accuracy: 
F1: 
Cosine-similarity: 
Num chunks retrieved: 
Notes:  not marked as correct/incorrect 
**Error type**: Retrieval: failure 

# Question: 53 (background) 
Query rewriting: correct
Generated answer:  incorrect 
Doc #1: incorrect 
Chunk #1: 
Section #1: 
EM Accuracy: 0.07092198581560283
Manual accuracy: 0.0
F1: 0.07092198581560283
Cosine-similarity: 0.616939902305603
Num chunks retrieved: 
Notes: possibly system has difficulty retrieving the background wiki docs, so have to use the recipes, which likely do not contain this type of info 
**Error type**: Retrieval: failure 

# Question: 54 (background) 
Query rewriting: correct 
Generated answer: incorrect (correct format) 
Doc #1: 
Chunk #1: 
Section #1: 
EM Accuracy: 0.0
Manual accuracy: 0.0
F1: 0.0
Cosine-similarity: 0.12016752362251282
Num chunks retrieved: 7
Notes:  see notes ^
**Error type**: Retrieval: failure 

# Question: 55 (background) 
Query rewriting: correct 
Generated answer: ‘the context does not contain the required information’
Doc #1: 
Chunk #1: 
Section #1: 
EM Accuracy: 
Manual accuracy: 
F1: 
Cosine-similarity: 
Num chunks retrieved: 
Notes:  # Question not used (not correct/incorrect) 
**Error type**: Retrieval: failure 


# Question: 56 (background)  
Query rewriting: correct 
Generated answer: correct 
Doc #1: not gold doc id, but answer could be found in doc
Chunk #1: -
Section #1: -
EM Accuracy: 0.33333333333333337
Manual accuracy: 1.0
F1: 0.33333333333333337
Cosine-similarity: 0.787550687789917
Num chunks retrieved: 9
Notes:
**Error type**: Retrieval: failure   

# Question: 57 (background) 
Query rewriting: correct
Generated answer: incorrect
Doc #1: incorrect 
Chunk #1: 
Section #1: 
EM Accuracy: 0.16494845360824742
Manual accuracy: 0.0
F1: 0.16494845360824742
Cosine-similarity: 0.36794427037239075
Num chunks retrieved: 10
Notes:  see ^ notes
**Error type**: Retrieval: failure 

# Question: 58 (background) 
Query rewriting: correct
Generated answer: partially correct (correct format) 
Doc #1: incorrect 
Chunk #1: 
Section #1: 
EM Accuracy: 0.16666666666666666
Manual accuracy: 0.428571428571
F1: 0.16666666666666666
Cosine-similarity: 0.5561586618423462
Num chunks retrieved: 3
Notes:  see ^ notes
**Error type**: Retrieval: failure 

###### Ignored 
# Question: 59 (multi-turn)
Query rewriting: 
Generated answer: 
Doc #1: 
Chunk #1: 
Section #1: 
EM Accuracy: 
Manual accuracy: 
F1: 
Cosine-similarity: 
Num chunks retrieved: 
Notes:  see notes (use of ‘text’ instead of ‘# Question’), cannot use

# Question: 60 (ingredients) 
Query rewriting: correct
Generated answer: incorrect (correct format) 
Doc #1: incorrect (at #5 so in top-5 k) 
Chunk #1: incorrect
Section #1: correct
EM Accuracy: 0.0
Manual accuracy: 0.0 
F1: 0.0
Cosine-similarity: 0.4317449927330017
Num chunks retrieved: 8 
Notes: answer would be correct if it was a different recipe (so correct response, but incorrect retrieval) 
**Error type**: retrieval: document 

# Question: 61 (general)
Query rewriting: correct 
Generated answer: incorrect
Doc #1: incorrect (but top-2)
Chunk #1: incorrect
Section #1: correct
EM Accuracy: 0.0909090909090909
Manual accuracy: 0.0
F1: 0.0909090909090909
Cosine-similarity: 0.22430986166000366
Num chunks retrieved: 
Notes: incorrect generation 
**Error type**: Retrieval: chunk, Generation: QAM

# Question: 62 (instructions) 
Query rewriting: correct 
Generated answer: incorrect generation 
Doc #1: correct 
Chunk #9: (1:9, 2:10, 3: info, 4: ingredients, 5: wr, 6: 0) 2/9
Section #1: correct
EM Accuracy: -
Manual accuracy: 0.0
F1: -
Cosine-similarity: -
Num chunks retrieved: 6
Notes:  did not retrieve all relevant documents, generation is completely wrong 
**Error type**: Retrieval: chunk, generation: QAM

# Question: 63 (general) 
Query rewriting: correct
Generated answer: correct 
Doc #1: correct
Chunk #1: correct
Section #1: correct
EM Accuracy: 0.6666666666666666
Manual accuracy: 1.0
F1: 0.6666666666666666
Cosine-similarity: 0.531109631061554
Num chunks retrieved: 7
Notes: correct generation, slightly different wording
**Error type**: None 

# Question: 64 (ingredients) 
Query rewriting: correct
Generated answer: correct 
Doc #1: correct 
Chunk #1: not golden, but info also found in that chunk (golden #5)
Section #1: not golden, but info also found in that chunk 
EM Accuracy: 0.0
Manual accuracy: 1.0
F1: 0.0
Cosine-similarity: 0.26012563705444336
Num chunks retrieved: 10
Notes: correct answer, formulated differently → lower accuracy   
**Error type**: None 

# Question: 65 (ingredients) (binary) 
Query rewriting: correct 
Generated answer: correct
Doc #1: correct 
Chunk #1: correct 
Section #1: correct 
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 3
Notes: correct  
**Error type**: None 

# Question: 66 (instructions) 
Query rewriting: correct 
Generated answer: incorrect 
Doc #1: incorrect 
Chunk #1: incorrect 
Section #1: incorrect
EM Accuracy: 0.0
Manual accuracy: 0.0
F1: 0.0
Cosine-similarity: 0.052024148404598236
Num chunks retrieved: 2
Notes: correct recipe (and chunk) never retrieved) 
**Error type**: retrieval: Doc

# Question: 67 (ingredients)
Query rewriting: correct
Generated answer: correct 
Doc #1: correct
Chunk #1: correct
Section #1: correct
EM Accuracy: 0.2222222222222222
Manual accuracy: 1.0
F1: 0.2222222222222222
Cosine-similarity: 0.19288265705108643
Num chunks retrieved: 2
Notes:  -
**Error type**: None 

# Question: 68 (ingredients) 
Query rewriting: correct 
Generated answer: correct 
Doc #1: correct
Chunk #1: incorrect (but #3 so top-3)
Section #1: incorrect
EM Accuracy: 0.0
Manual accuracy: 1.0
F1: 0.0
Cosine-similarity: 0.7942878007888794
Num chunks retrieved: 10
Notes:  correct answer, but formulated different than expected answer 
**Error type**: None 

# Question: 69 (ingredients) 
Query rewriting: correct 
Generated answer: partially correct (asks for grams, but first answers with tbsp)
Doc #1: incorrect (but #2)
Chunk #1: incorrect (but #3)
Section #1: incorrect 
EM Accuracy: 0.0
Manual accuracy: 0.5
F1: 0.0
Cosine-similarity: 0.5215567350387573
Num chunks retrieved: 8
Notes:  partially correct, addition of irrelevant info (was not asked for) 
**Error type**: Generation: addition 

# Question: 70 (general) 
Query rewriting: correct 
Generated answer: incorrect 
Doc #1: (1: no, 2: no, 3: no, 4: no,) incorrect 
Chunk #1: -
Section #1: -
EM Accuracy: -
Manual accuracy: 0.0
F1: -
Cosine-similarity: -
Num chunks retrieved: 4
Notes:  never retrieved documents that satisfied this constraint 
**Error type**: Retrieval: doc 

# Question: 71 (ingredients)
Query rewriting: correct 
Generated answer: correct
Doc #1: correct
Chunk #1: incorrect (but #2)
Section #1: incorrect
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 0.9999999403953552
Num chunks retrieved: 10
Notes: correct   
**Error type**: None 

###### Ignored 
# Question: 72 (multi-turn)  
Query rewriting: 
Generated answer: 
Doc #1: 
Chunk #1: 
Section #1: 
EM Accuracy: 
Manual accuracy: 
F1: 
Cosine-similarity: 
Num chunks retrieved: 
Notes: see ^ notes

# Question: 73 (ingredients)
Query rewriting: correct 
Generated answer: correct but could have been longer (say with what cheese, but technically it is correct) 
Doc #1: correct 
Chunk #1: incorrect (it is in steps but in #3)
Section #1: incorrect 
EM Accuracy: 0.0
Manual accuracy: 1.0
F1: 0.0
Cosine-similarity: 0.10666216164827347
Num chunks retrieved: 10
Notes:  technically correct, but could have added more context
**Error type**: Generation: omission details 

# Question: 74 (ingredients) 
Query rewriting: correct 
Generated answer: correct
Doc #1: correct
Chunk #1: correct
Section #1: correct
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 3
Notes:  -
**Error type**: None 

# Question: 75 (general) 
Query rewriting: correct 
Generated answer: correct 
Doc #1: not golden, but inside also says what kind of knife you need for oysters (but specific recipe never retrieved)
Chunk #1: not golden but ^
Section #1: correct
EM Accuracy: 0.3
Manual accuracy: 1.0
F1: 0.3
Cosine-similarity: 0.6969315409660339
Num chunks retrieved: 4
Notes:  correct answer, but recipe that was mentioned in # Question was never retrieved
**Error type**: Retrieval: doc

# Question: 76 (general/instructions)  
Query rewriting: correct
Generated answer: correct 
Doc #1: incorrect (but #3 so top-3)
Chunk #1: incorrect (but #3 so top-3)
Section #1: correct
EM Accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 5
Notes: correct
**Error type**: None 

# Question: 77 (general)
Query rewriting:correct 
Generated answer: partially correct, mentions the word ‘variations’, but doesn’t give one, the recipe talked about is also a variation of lasagna 
Doc #1: (1: yes, 2: yes, 3: yes, 4: no, 5: yes, 6: yes, 7 yes, 8: yes, 9, yes) 8/9 retrieved docs were lasagna 
Chunk #1: -
Section #1:- 
EM accuracy: -
Manual accuracy: 0.25
F1: -
Cosine-similarity: -
Num chunks retrieved: 9
Notes: gives a variation of lasagna, but phrasing/wording is not what you would expect from such a # Question. Most documents retrieved were lasagna variations 
**Error type**: Generation: Incomplete Answer

# Question: 78 (general) 
Query rewriting: correct 
Generated answer: correct 
Doc #1: correct 
Chunk #1: correct 
Section #1: correct 
EM accuracy: 0.0
Manual accuracy: 1.0
F1: 0.0
Cosine-similarity: 0.14403444528579712
Num chunks retrieved: 10
Notes: low accuracy, due to additional info in expected answer (which is not necessarily needed)
**Error type**: None 

# Question:  79 (instructions) 
Query rewriting: correct 
Generated answer: correct
Doc #1: correct 
Chunk #1: correct 
Section #1: correct
EM accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0 
Cosine-similarity: 1.0
Num chunks retrieved: 10
Notes: -
**Error type**: None 

# Question: 80 (ingredients)
Query rewriting: correct 
Generated answer: partially (correct format, partially correct answer (need both, only mentions 1)) 
Doc #1: correct 
Chunk #1: incorrect  (but #3)
Section #1: incorrect
EM accuracy: 0.0
Manual accuracy: 0.5
F1:  0.0
Cosine-similarity: 0.3551669120788574
Num chunks retrieved: 9
Notes: omission black pepper
**Error type**: Generation: omission 

###### Ignored 
# Question: 81 (instructions/general)
Query rewriting: correct 
Generated answer: "The context does not contain the required information."
Doc #1: correct 
Chunk #1: incorrect (never retrieved) 
Section #1: correct
EM accuracy: 
Manual accuracy:
F1:
Cosine-similarity: 
Num chunks retrieved:
Notes: # Question ignored, but should have been able to figure out if retrieved correct chunk (CHANGE in doc GOLD step is 1 not 0)

# Question: 82 (ingredients/general) 
Query rewriting: correct 
Generated answer: correct 
Doc #1: correct 
Chunk #1: not golden, but technically also there 
Section #1:
EM accuracy: 0.0
Manual accuracy: 1.0
F1: 0.0 
Cosine-similarity: 0.3909313976764679
Num chunks retrieved: 10 
Notes: has tag seafood, correct 
**Error type**: None 

###### Ignored 
# Question: 83 (multi-turn) 
Query rewriting:
Generated answer:
Doc #1:
Chunk #1:
Section #1:
EM accuracy: 
Manual accuracy:
F1:
Cosine-similarity: 
Num chunks retrieved:
Notes: ignore, see ^ notes 

###### Ignored 
# Question: 84 (general) 
Query rewriting: correct 
Generated answer: “context does not contain required info” 
Doc #1: (1: correct (incorrect), 2: correct (incorrect), 3: no, 4: no, 5: yes, 6: yes, 7: no, 8: no, 9: no, 10: no) 2/10 
Chunk #1: - 
Section #1: - 
EM accuracy: 
Manual accuracy:
F1:
Cosine-similarity: 
Num chunks retrieved:
Notes: ignore # Question, note: some of the golden docs inside json do have coffee


# Question: 85 (ingredients) 
Query rewriting: correct 
Generated answer: correct 
Doc #1: correct
Chunk #1: correct 
Section #1: correct 
EM accuracy: 0.0
Manual accuracy: 1.0
F1: 0.0
Cosine-similarity: 0.21345362067222595
Num chunks retrieved: 10
Notes: - 
**Error type**: None 

# Question: 86 (ingredients) 
Query rewriting: correct 
Generated answer: correct 
Doc #1: correct 
Chunk #1: incorrect (#4) 
Section #1: incorrect 
EM accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 10
Notes: correct, gold chunk in retrieved top 5
**Error type**: None 

# Question: 87 (ingredients) 
Query rewriting: correct
Generated answer: correct 
Doc #1: correct 
Chunk #1: incorrect (but #2)
Section #1: incorrect 
EM accuracy: 0.6666666666666666
Manual accuracy: 1.0
F1: 0.6666666666666666
Cosine-similarity: 0.3578505516052246
Num chunks retrieved: 7 
Notes: correct, but slightly different wording 
**Error type**: None 

# Question: 88 (instructions) 
Query rewriting: correct 
Generated answer: correct 
Doc #1: correct 
Chunk #1: correct 
Section #1: correct 
EM accuracy: 0.56
Manual accuracy: 1.0
F1: 0.56 
Cosine-similarity: 0.8234919309616089
Num chunks retrieved: 10
Notes: accuracy not exactly one due to slight difference 
**Error type**: None 

# Question: 89 (general) 
Query rewriting: correct 
Generated answer: correct 
Doc #1: correct 
Chunk #1: correct 
Section #1: correct 
EM accuracy: 1.0
Manual accuracy: 1.0
F1: 1.0
Cosine-similarity: 1.0
Num chunks retrieved: 10
Notes: -
**Error type**: None 

# Question:  90 (instructions) 
Query rewriting: correct 
Generated answer: incorrect, lists ingredients instead of explaining, but correct recipe 
Doc #1: correct
Chunk #3: incorrect (but #3, #5) missing 1 correct chunk 
Section #1:
EM accuracy: -
Manual accuracy: 0.0
F1: -
Cosine-similarity: -
Num chunks retrieved: 10
Notes: not all correct chunks were retrieved, # Question was answered incorrectly 
**Error type**: Generation: QAM, Filter: wrong tag  

# Question:  91 (general) 
Query rewriting: correct 
Generated answer: (partially) correct (it is ‘very’ difficult, not just ‘difficult) 
Doc #1: correct 
Chunk #1: correct 
Section #1: correct 
EM accuracy: 0.0
Manual accuracy: 0.75
F1: 0.0
Cosine-similarity: 0.17471644282341003
Num chunks retrieved: 10
Notes: almost correct generation, correct retrieval 
**Error type**: Generation: omission detail 

# Question:  92 (general) 
Query rewriting: correct 
Generated answer: incorrect (correct format, incorrect answer) 
Doc #1: (1: no, 2: no)
Chunk #1:
Section #1:
EM accuracy: - 
Manual accuracy: 0.0
F1: -
Cosine-similarity: - 
Num chunks retrieved: 2
Notes: no relevant recipes retrieved
**Error type**: Retrieval: doc 

###### Ignored 
# Question:  93 (general) 
Query rewriting: correct 
Generated answer: “context does not contain” 
Doc #1: (1: no, 2: no, 3: no, 4: no, 5: no, 6: no, 7: no)
Chunk #1:
Section #1:
EM accuracy: 
Manual accuracy:
F1:
Cosine-similarity: 
Num chunks retrieved:
Notes: ignore # Question, should be able to answer, misunderstood do not like (as many retrieved recipes contain chocolate or are tiramisu) 

# Question:  94 (general) 
Query rewriting: correct 
Generated answer: partially correct (technically it is right, but it is expected that it would recommend a recipe) 
Doc #1: (1: yes) 
Chunk #1:
Section #1:
EM accuracy:  -
Manual accuracy: 0.25
F1: -
Cosine-similarity:  -
Num chunks retrieved:
Notes: able to retrieve a relevant doc, answer correct, but expected to have more information 
**Error type**: Generation: Incomplete answer/omission 

###### Ignored 
# Question:  95 (multi-turn → ignore) 
Query rewriting:
Generated answer:
Doc #1:
Chunk #1:
Section #1:
EM accuracy: 
Manual accuracy:
F1:
Cosine-similarity: 
Num chunks retrieved:
Notes: 

# Question:  96 (instructions) 
Query rewriting: correct 
Generated answer: incorrect, starts to explain one of the steps, but it should mention not knowing a recipe 
Doc #1: -
Chunk #1: -
Section #1: -
EM accuracy: -
Manual accuracy: 0.0
F1: -
Cosine-similarity: - 
Num chunks retrieved:
Notes: should have mentioned not being sure/not knowing a recipe 
**Error type**: Generation/retrieval: failure to abstain

# Question: 97 (instructions) 
Query rewriting: correct 
Generated answer: incorrect, starts listing ingredients 
Doc #1: correct 
Chunk #1: incorrect 
Section #1: incorrect 
EM accuracy: -
Manual accuracy: 0.0
F1: -
Cosine-similarity:  -
Num chunks retrieved: 1
Notes: got tag ingredients, but was an instruction # Question → answered incorrectly 
**Error type**: Filter: wrong tag → Generation: QAM

# Question: 98 (general)
Query rewriting: correct 
Generated answer: (partially) correct format, starts explaining first step of wrong recipe
Doc #1: incorrect 
Chunk #1: incorrect 
Section #1: incorrect 
EM accuracy: -
Manual accuracy: 0.0
F1: - 
Cosine-similarity: -
Num chunks retrieved: 1
Notes: wrong tag of ‘dessert’, wrong chunk/doc retrieved → wrong answer 
**Error type**: Filter: wrong type, Retrieval: doc

###### Ignored 
# Question:  99 (instructions) 
Query rewriting: incorrect (starts mentioning green tea and chocolate cake) 
Generated answer:
Doc #1:
Chunk #1:
Section #1:
EM accuracy: 
Manual accuracy:
F1:
Cosine-similarity: 
Num chunks retrieved:
Notes: ignore, most likely problem with eval cell

###### Ignored 
# Question: 100 (multi-turn) → ignore
Query rewriting:
Generated answer:
Doc #1:
Chunk #1:
Section #1:
EM accuracy: 
Manual accuracy:
F1:
Cosine-similarity: 
Num chunks retrieved:
Notes: 


# Question: 
Query rewriting:
Generated answer:
Doc #1:
Chunk #1:
Section #1:
EM accuracy: 
Manual accuracy:
F1:
Cosine-similarity: 
Num chunks retrieved:
Notes: 


