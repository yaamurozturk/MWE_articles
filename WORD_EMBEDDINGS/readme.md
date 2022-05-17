Main question: Can meaning differences among MWEs be estimated using word embeddings?
---

- to measure distance between clusters we need to compare the distance between two individuals (so two occurrences of mwes)

- bert: if we look in terms of meaning, abstract representation of meaning of an expression, compare to meaning of a mwe in another sentence

- or non-contextual could be direct representation of types and contextual would be occurrences

Do we want to work on identification of mwes using word embeddings, or extraction or what? and why do we want to calculate the distance among different types of mwes, how will this help the task? 
do we want to deduce the meaning of non-compositional mwes with word embeddings, is it possible that pay_attention will be deemed similar to focus. 
pay attention -> focus, homework…
kick the bucket -> funeral, dead, bad… 

1. you work on how to compute semantic distances between two types, this means understanding word embedding, and trying to figure out how we can use word embedding for MWE (they usually only work for simple words). On that subject I don't know much so I can't really tell you where to start your readings (I can probably still help you find some interesting paper but we'd need to discuss about it first) 

2. The Distributional Hypothesis is that words that occur in the same contexts tend to have similar meanings [2]. -> can be tested for mwes. 
working on bigrams is a possible way of capturing mwes but we can miss most of the constructions, and have many meaningless ones that simply occur as bigrams. maybe for vmwes, we can filter this by getting bigrammes that have at least one verb lemma, which can be comparable with the parseme corpora. (but this is not discontinuity friendly/set window to 2 or three) (pay special attention) 
bigram approach might only attribute similar bigrams to our mwe bigrams which is not what we want, meaning can be similar to one single token such as pay attention 
For this case i suggest we only work on LVCs and VIDs, actually VPCs can also be interesting (turn off, make out) seems to be difficult to come up with similarity. For vpcs we can use a bigram method. 

3. you work on how with a matrix pairwise distance between types we can compute a disparity score. On that side of things An overview of the Weitzman approach to diversity is an interesting read

Word2vec
---
word2vec - glove for word representation stanford lecture
word2vec works with a skip-gram model, you hvae a word and you take one window at a time to guess the word. 

question: is the possibility of mwe components higher to occur together than other similar word. components’ simiarity can be quite low (spill and beans e.g) in other context. so how do we identify if the usage of these words are idiomatic. 
-> we need to establish the reason why we want to work on word embeddings 

we have a corpus annotated for mwes. the skipped ones in the consistency check can work as literal occurrences (after double annotation for example to check if its definetely the right usage or not) 

what can be done as an experiments is to compare the context of mwes with their own literal occurrences. so we take a corpus, we extract all the occurrences of a VID 

- predictive embedding model, predicts most likely word in the given context
-uses 2 types of architecture:
    CBOW (continuous bag of words)
-words that have equal probability of occurring in a given context is deemed similar
    Skip-gram
-works the other way round from cbow
-predicts the context according to the given word

shortcomings(?) 
-does not take the general context into account
-ignores the fact that some context words occur more than others

How would it work for MWEs?
Can work for mwe identification or extraction?
for constructions like LVCs rather than VIDs I think
Can word2vec skipgram guess the context of VIDs? (to swallow one’s pride, bitter pill to swallow, (be) on its way , 
the issue is idiomatic phrases occur much more rarely..
we can use skip gram for compositionality ratings. less similar the words to their context, more non compositional we can say (for examples like swallow pride, but it can be the opposite for (bitter pill to swallow, kick the bucket…)

-Window based methods suffer from the disadvantage that they do not operate directly on the co-occurrence stats of the corpus, instead they scan context windows across the entire corpus and fail to take advantage of the vast amount of repetition in the data. (glove paper) 
They basically establish two categories as Shallow Window Based methods and Matrix Factorization Methods. 

GloVe
---
-aggregated global word to word co-occurrence matrix from a given collection of texts, denser and more expressive vector representation.
-tested with analogies and different sizes of corpus, pointed out that the system does not necessarily work better on bigger corpus.
-they are talking about billions of tokens while training, i dont know how this would work on mwes 

Ideas: 
for cross lingual works is there a way of knowing if mwes are mostly translated as mwes or single words?? does it depend on a language, but for this we need parallel corpus. we can parse a parallel corpus and extract mwes that exist in parseme?? 



Summaries for articles 
---
1. Pre-tokenization of Multi-word Expressions in Cross-lingual Word Embeddings (Naoki Otani, Satoru Ozaki, Xingyuan Zhao, Yucen Li, Micaelah St Johns, Lori Levin)
-Cross lingual word embeddings (close vector representation of words that have similar meanings regardless of language) 
-dealing with mwes using word embeddings is a bit problematic because each component of an mwe gets its own vector, therefore losing the meaning of the full component. Basically, MWEs are not translated by CWEs. 
- In this article they propose a method of previously tokenizing the mwes gathered from a list of mwes. 
-They show how the pre-tokenization of mwes as a single token performs better than averaging the embeddings of individual components of the mwe. 

-Why averaging the vectors does not work? because of non-compositionality obviously. 

-they give the example of alignment of “United” and “States” in chinese and english. Indeed united_states as a single token aligns more closer to the single token that means US in chinese. 
-it is also pointed out that single token vectors of “united” and “states” is much closer than they should be, even though the separate meanings are not that similar. so since they are used mostly together, it confuses the vector representation. (this can also be the case for other mwes like verbal ones that we use a lot, of course won’t be as much as US but idk..)
-They use a lexicon based approach to identify mwes in a corpus because automatic methods are proven to be still problematic. The dataset is in ten language pairs and contains MWEs in addition to single orthographic tokens. Shortcoming of lexicon approach is that you cannot identify mwes in the corpus that do not exist in the lexicon. 
-monolingual word embeddings: fasttext with cbow. trained mwes are taken as one token “french_fries” has a different vector from those of “french” and “fries”.
-cross-lingual mapping of embeddings: they take two sets of word embeddings from different languages and align the source em. to target em. based on a bilingual dictionary. 
evaluating translation: 
-In terms of MWE types, compound (c) was the easiest category to translate (success rate of 60.22%), and flat+fixed+idiom (ffi), which includes various idiomatic expressions, was the hardest (25.52%)
-In terms of parts-of-speech of MWEs, it turned out that verbal MWEs were much more difficult to translate (21.01%) than nominal MWEs (48.06%).
-adverbials translated with good accuracy, might be because the variety of context they are used is not very big
-easiest to translate are noun compounds 
-While stop words such as “in” and “a” are usually not aligned with significant words, the inclusion of these words in MWEs (e.g., in vain and a bit) establishes meaningful relationships across languages. 

This article proves that taking mwes as single tokens might work for processing of mwes. 
—-----------------------------

2. A Single Word is not Enough: Ranking Multiword Expressions Using Distributional Semantics (Martin Riedl, Chris Biemann)
-they propose a mechanism that ranks n-grams.
-they introduce a new concept to describe the multiwordness of a term by its uniqueness, which represents the likeliness of a term to be replaced with one token 
-hypothesis: n-grams, which are MWE, could be substituted by single words, thus they have many single words amongst their most similar terms. When a semantically non-compositional word combination is added to the vocabulary, it expresses a concept that is necessarily similar to other concepts. Hence, if a candidate multiword is similar to many single word terms, this indicates multiwordness.


3. Unsupervised multilingual word embeddings (mwe here refers to multlingual word embeddings) Xilun Chen, Claire Cardie

multilingual word embeddings represent words from multiple languages in a single distributional vector space. 
-> meaning that words from different languages with similar meanings will be closely represented as vectors
-> bilingual word embeddings connects the lexical semantics of two languages, to train these, cross-lingual supervision is required (parallel corpora or bilingual lexica) so it is a bit costly and not low resource language friendly
-> used fasttext
-> they say that syntactic similarity between languages help with the quality of MWE,obviously. they worked on a few germanic and romance languages

4. SPINE: Sparse Interpretable neural embeddings
- aiming to show more interpretable results than glove and word2vec
- interpretable meaning that the top participating words do not form a semantically coherent group
- they made a qualitative analysis, 
idea: can compare previous vectors (glove and word2vec) with SPINE vectors. 
idea2: two different matrices can be compared, one with tokenizing mwes as one and one separately to see meaning shift in words (which will point out to compositionality) 


code git downloaded to SPINE directory, can be tested and qualitative analysis is also available for this model. 


-idea3: testing the “word translation without parallel data” on mwes. (https://arxiv.org/pdf/1710.04087.pdf) 

- it works better on most frequent words ofc, so they train on most frequent 50k words, this can be a problem to adapt to mwes bcs the light verbs have probably higher frequency than other components.  

- a generic Framework for Multiword Expressions Treatment:
from Acquisition to Application https://aclanthology.org/W12-3311.pdf (explains very well and briefly, the problem of mwe processing)
Integrating Word Embeddings in the
mwetoolkit for Semantic MWE Processing

- mwetoolkit (https://aclanthology.org/L16-1194.pdf) /to come back later
integrating word embeddings for semantic mwe processing


About disparity and word embeddings
---
- can try the model SPINE and word2vec  (why spine is good, it gives more interpretable results than glove and word2vec) 
- can start with german and french (due to accurate lemmas and german has one token mwes, which can be helpful in the case of word embeddings) 
- can use the raw corpora in parseme, already annotated
- can use MTLB-struct for identifying mwes in the raw corpora.  (and TRAVIS )
- pre-tokenizing mwes is a problem for discontinuous mwes (maybe what we can do is
Word2vec and Glove word embeddings are context independent- these models output just one vector (embedding) for each word, combining all the different senses of the word into one vector.
That is the one numeric representation of a word (which we call embedding/vector) regardless of where the words occur in a sentence and regardless of the different meanings they may have. For instance, after we train word2vec/Glove on a corpus (unsupervised training - no labels needed) we get as output one vector representation for, say the word “cell”. So even if we had a sentence like “He went to the prison cell with his cell phone to extract blood cell samples from inmates”, where the word cell has different meanings based on the sentence context, these models just collapse them all into one vector for “cell” in their output.
ELMo and BERT can generate different word embeddings for a word that captures the context of a word - that is its position in a sentence.
For instance, for the same example above “He went to the prison cell with his cell phone to extract blood cell samples from inmates”, both Elmo and BERT would generate different vectors for the three vectors for cell. The first cell (prison cell case) , for instance would be closer to words like incarceration, crime etc. whereas the second “cell” (phone case) would be closer to words like iphone, android, galaxy etc.. This can work better in examples of mwes with polysemy such as (make up)
The main difference above is a consequence of the fact Word2vec and Glove do not take into account word order in their training - ELMo and BERT take into account word order (ELMo uses LSTMS; BERT uses Transformer - an attention based model with positional encodings to represent word positions).
A practical implication of this difference is that we can use word2vec and Glove vectors trained on a large corpus directly for downstream tasks. All we need is the vectors for the words. There is no need for the model itself that was used to train these vectors.
However, in the case of ELMo and BERT, since they are context dependent, we need the model that was used to train the vectors even after training, since the models generate the vectors for a word based on context. We can just use the context independent vectors for a word if we choose too (just get the raw trained vector from the trained model) , but this would defeat the very purpose/advantage of these models. 


------------------------------------------------------------------------------------------

word2vec/GloVe                            Bert/Elmo

-word order not important                        word order important 
-one vector for the same word    
different vectors for the same word that occur in different contexts 


-----------------------------------------------------------------------------------------

-> stuff to keep in mind: discontinuity, general preprocessing and mwe preprocessing 
-> if we use raw files, we need to annotate for mwes. 
-> trying first with the small data of french. 
-> using already lemmatized and word forms can be compared, to see the meaning shift  
-> non contextual we
