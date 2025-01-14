# Week 2: Natural Language Processing & Word Embeddings

- [Introduction to Word Embeddings](#introduction-to-word-embeddings)
    - [Word Representation](#word-representation)
    - [Using word embeddings](#using-word-embeddings)
    - [Properties of word embeddings](#properties-of-word-embeddings)
    - [Embedding matrix](#embedding-matrix)
- [Learning Word Embeddings: Word2vec & GloVe](#learning-word-embeddings-word2vec--glove)
    - [Learning word embeddings](#learning-word-embeddings)
    - [Word2Vec](#word2vec)
    - [Negative Sampling](#negative-sampling)
    - [GloVe word vectors](#glove-word-vectors)
- [Applications using Word Embeddings](#applications-using-word-embeddings)
    - [Sentiment Classification](#sentiment-classification)
    - [Debiasing word embeddings](#debiasing-word-embeddings)


NLP w/ deep learning form a powerful pair for advanced text processing. Using Word vector representations and embedding layers, you can train RNNs with exceptional performance in various NLP tasks. Applications include -- *sentiment analysis, named entity recognition and machine translation*.


## Introduction to Word Embeddings

### Word Representation

**Context & Motivation**
- **Past Focus** was on RNNs, GRUs, and LSTMs.  

- **Current Focus** is on i) exploring Natural Language Processing (NLP), which has been transformed by deep learning techniques and ii) introducing word embeddings, a key technique for capturing word relationships and similarities.  

**Traditional Word Representation and its Limitations**

- Words were traditionally represented using one-hot vectors, where each word corresponds to a unique binary vector.  
- One-hot representation treats each word independently as isolated entities, preventing algorithms from generalizing effectively across words.
- The dot product of any two distinct one-hot vectors is zero, meaning they cannot capture semantic relationships, such as "apple" and "orange" being more similar than "king" and "orange" or "queen" and "orange."

**Desire for Featurized Representation** 

- **Proposed Solution:** Represent words using features like gender, royalty, age, or type (e.g., food vs. non-food). The goal is to have similar feature vectors for similar words.  

**High-Dimensional Embeddings**  
- Instead of just a few features, words can be embedded in a high-dimensional space (e.g., 300 dimensions) for richer representation and better generalization across words.  
- Words like “apple” and “orange” would have similar embeddings due to shared features (e.g., both are fruits).  

**Visualizing Embeddings**

- Although embeddings can have high-demsions, they can be visualized in 2D or 3D using techniques like t-SNE.  
- Visualization often shows that similar words cluster together.

  ![word-embedding](images/word-embedding.png)

**What is an Embedding?**  
- An embedding maps each word to a point in a high-dimensional space based on its learned features.  


**Significance of Word Embeddings** 
- Word embeddings revolutionized NLP by enabling algorithms to understand word relationships and similarities.  
- They provide a robust foundation for many NLP tasks. 


### Using word embeddings

- Word embeddings enable algorithms to generalize across scenarios by understanding word similarities.  
- Example: Training on “Sally Johnson is an orange farmer” helps the model generalize to “Robert Lin is an apple farmer” due to the similarity between "orange" and "apple."  
- Even rare phrases like “Robert Lin is a durian cultivator” can be understood as referring to a person, leveraging the embeddings' semantic understanding of “durian” and “cultivator.”

- **Learning Word Embeddings** Embeddings are learned from large amounts of unlabeled text, often from the internet.  

**Transfer Learning in Word Embeddings**  
- Transfer learning allows embeddings learned from large corpora to be applied to new NLP tasks with smaller datasets.  
- Fine-tuning embeddings on the new task's dataset can further enhance performance, especially with larger datasets.

**Applications of Word Embeddings**  
- Useful for NLP tasks like:  
  - Named Entity Recognition  
  - Text Summarization  
  - Co-reference Resolution  
  - Parsing  
- For tasks with large dedicated datasets (e.g., language modeling, machine translation), embeddings may offer less advantage.

**Comparison with Face Encoding**  
- Word embeddings are similar to face encoding in image recognition:  
  - **Face Encoding:** Encodes any face image dynamically.  
  - **Word Embeddings:** Learns fixed vector representations for a predefined vocabulary.

**Key Takeaway**  
- Word embeddings are dense vector representations capturing semantic relationships between words.  
- They improve generalization across contexts, enabling better performance in NLP tasks, even with smaller labeled datasets.  


### Properties of Word Embeddings

- **Reasoning Through Analogies:** Word embeddings can infer relationships in analogies, such as “man is to woman as king is to queen.”  
  - Words are represented as vectors (e.g., $e_{\text{man}}$, $e_{\text{woman}}$).  
  - Differences between vectors, like $e_{\text{man}} - e_{\text{woman}}$, often represent specific features (e.g., gender).  
  - The algorithm uses these patterns to find missing words in analogies.  

- The most common metric to measure similarity between word embeddings is cosine similarity.  
- Calculates the cosine of the angle between two vectors to determine their closeness.  

**Examples of Captured Relationships**  
- *Gender Relationships:* Man to Woman :: Boy to Girl  
- *Geography:*  Ottawa to Canada :: Nairobi to Kenya  
- *Comparatives:*  Big to Bigger :: Tall to Taller  
- *Currency and Country:*  Yen to Japan :: Ruble to Russia  

**Conclusion**  
- Word embeddings capture rich semantic relationships and reasoning abilities.  
- Analogies provide insight into their strength, even if not a direct application.  
- Understanding how embeddings are learned from data is the next step in advancing NLP.  

### Embedding Matrix
- When we implement algorithm to learn a word embedding, what you end up learning is an embedding matrix.
  - Let $E$ be an embedding matrix of size (300, 10000), where first dimension is embedding dimension while the second dimension is the size of vocabulary.
  - $O_{6257} = [0,......0,1,0,...,0]$ be OHE of 6357th word in vocabulary of size (10000, 1)
  - $E·O_{6257} = e_{6257}$ gives us embedding of 6357th word with dimension (300, 1)

- Our goal will be to learn an embedding matrix $E$ by initializing it randomly and then learning all the parameters of this (300, 10000) dimensional matrix.
- $E$ times the one-hot vector gives you the embedding vector.
- In practice, use specialized function to look up an embedding.


## Learning Word Embeddings: Word2vec & GloVe

- Initially, complex algorithms were used to develop word embeddings.  
- Simpler and more efficient methods, like Word2Vec, emerged over time.  
- Understanding complex models first can help build intuition before transitioning to simpler approaches.  

- **Foundation of Word Embeddings: Neural Language Model** 
    - The model predicts the next word in a sentence (e.g., predicting “juice” after “I want a glass of orange”).  
  - Each word is represented as a one-hot vector, which is multiplied by an embedding matrix ($E$) to generate its compact representation.  
  - These embeddings are fed into a neural network to predict the next word using a softmax layer.  

- **Training and Context:**  
  - During training, the model learns associations based on observed data, such as frequently predicting “juice” after “orange.”  
  - *To handle varying sentence lengths*, the model uses a fixed number of preceding words (e.g., last four) as input.  

- **Importance of Word Embeddings:**  
  - Words in similar contexts (e.g., “apple” and “orange”) have similar embeddings.  
  - This similarity allows the model to generalize and learn effectively from data.  

- **Defining Context for Predictions:**  
  - Context can include preceding words, both preceding and succeeding words, or a random nearby word (e.g., Skip-Gram model).  
  - The choice of context depends on the task, such as language modeling or word embedding learning.  

- **Conclusion:**  
  - Word embeddings use contextual information to create meaningful word representations.  
  - Simpler models like Word2Vec are effective and highlight the power of embeddings in capturing word relationships.  


### Word2Vec
- Is a simple and more efficient algorithm for learning word embeddings compared to other neural language models.
- Paper: [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) by Tomas Mikolov, Kai Chen, Greg Corrado, Jeffrey Dean.

**Skip-Gram Model**

- It creates a supervised learning problem using context-to-target pairs from sentences. 

- Rather than having a fixed number of words, a random word is selected from the sentence as the context (say 'orange').
- Another random word, within a specified window ($\pm5$ or $\pm10$) is selected as the target (say 'juice', 'glass', 'my').

    | Context | Target |
    | :----: | :----: |
    | orange | juice |
    | orange | glass |
    | orange | my |

- **Objective**: Predict a random word within $\pm5$ or $\pm10$ word window of the input context word.

- In fact, the objective isn't necessarily to perform perfectly on this problem, but to use it as a method to learn good word embeddings.

**Model details**

- Context `c`: `orange` and target `t`: `juice`.

- A little neural network that basically looks up the embedding and apply a softmax unit.

    $o_c \rightarrow E$  ---> $e_c\rightarrow$  $O$(softmax) $\rightarrow \hat{y}$ 

- **Softmax**: 

    $$\dfrac{e^{\theta_t^{T}e_c}}{\sum\limits_{j = 1}^{10000}e^{\theta_j^{T}e_c}}$$
    where, $\theta_t$ is a parameter associated with output `t`. (bias term is omitted)

- **Loss**: $L(\hat{y}, y) = -\sum\limits_{i = 0}^{E} y_i\log(\hat{y_i})$ 

- **Why skip-gram?** It takes one word as input/context and tries to predict some words skipping a few words from the left or the right side.

**Computational Concerns**
- The primary concern of the skip-gram model is **computational speed** due to the softmax step.
- Calculating the softmax requires summing over the entire vocabulary, which is computationally expensive, especially for large vocabularies.

**Hierarchical Softmax Classifier**
- One of a few solutions to address the computational problem w/ the skip-gram model.
- Instead of classifying among all words at once, a binary tree of classifiers is used to narrow down choices.
- This reduces the computational cost from linear to logarithmic with respect to vocabulary size.
- The tree’s structure can be optimized so that more frequent words are near the top, reducing traversal time for common words.

**Sampling Context Words**

- For training, context words aren’t always chosen uniformly from the corpus.
- Some words appear more frequently than others, and sampling uniformly might bias the training towards these common words.
- In practice, various heuristics are employed to balance sampling common and less common words to ensure that embeddings are learned for both.

**Continuous Bag-of-Words (CBOW)**

- The other version of the Word2Vec model is CBOW.
- The CBOW model predicts a middle word from its surrounding context, and it has its own advantages and disadvantages.



### Negative Sampling: An Efficient Approach to Learning Word Embeddings

- Negative sampling is a computationally efficient modification of the Skip-Gram model for learning word embeddings.
- Introduced in the paper [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) by Tomas Mikolov et al.

**Core Idea**
- For a given pair of words, such as "orange" and "juice," the goal is to determine if they form a **context-target pair**:
  - **Positive Example:** "Orange" and "juice" (labeled 1).  
  - **Negative Examples:** "Orange" and randomly picked words from vocabulary like "king" or "book" (labeled 0).  

**Generating Training Data**
1. **Positive Examples:**  
   - Select a context word (e.g., "orange").  
   - Choose a target word from a window of words surrounding the context word (e.g., "juice").  
2. **Negative Examples:**  
   - Use the same context word ("orange") and randomly select $k$ unrelated words from the vocabulary.  

**Choosing $k$**
- $k$ represents the number of negative samples for every positive example:  
- In this example, k=4. x=`(context, word)`, y=`target`.

    | context | word | target? |
    | :----: | :----: | :----: |
    | orange | juice | 1 |
    | orange | king | 0 |
    | orange | book | 0 |
    | orange | the | 0 |
    | orange | of | 0 |

- **Smaller Datasets:** $k$ between 5 and 20 is recommended  
- **Larger Datasets:** $k$ between  2 and 5 is recommended for efficiency.  



**Training the Model**
- Negative sampling formulates a **binary classification problem**:
  - Predict if a word pair is contextually related (1) or not (0).  
- Instead of training a full Softmax layer with all vocabulary words (e.g., 10,000 words), negative sampling trains $k+1$ logistic regression classifiers per iteration, significantly reducing computational costs.  

**Choosing Negative Examples**
- Uniformly sampling negative words ($p(w) = \frac{1}{|V|}$) is inefficient, as it doesn't reflect real-world word distributions.  
- Sampling by empirical frequency over-represents common words like "the" or "and." 
- the authors proposed a heuristic: sample words proportionally to the frequency of a word raised to the power of 3/4. 
  
  $$p(w) \propto f(w)^{3/4}$$
  where $f(w)$ is the observed frequency of a word. This balances representation effectively.  

**Advantages of Negative Sampling**
- **Efficiency:** Reduces computational complexity by avoiding the need to update all weights in a large Softmax layer.  
- **Quality:** Produces high-quality word embeddings suitable for various NLP tasks.  
- **Flexibility:** Embeddings can be learned from scratch or initialized with pre-trained embeddings for quick deployment.  


### GloVe: Global Vectors for Word Representation

**Introduction**
- is another popular word embedding algorithm.
- differs from Word2Vec or skip-gram models
- provides a simple yet effective mechanism for learning embeddings based on word co-occurrences.  

**How GloVe Works?**
- **Co-occurrence Matrix ($X_{ij}$):**  
  - GloVe counts how often two words ($i$ and $j$) appear near each other in a corpus, within a defined window (e.g., ±10 words).  
  - $X_{ij}$ represents the number of times word $j$ appears in the context of word $i$.  

- **Optimization Objective:**  
  - GloVe minimizes the difference between the dot product of the embedding vectors for words $i$ and $j$ and the logarithm of their co-occurrence, i.e., $\log(X_{ij})$:

  $$\text{minimize}\sum\limits_{i = 0}^{10000}\sum\limits_{i = 0}^{10000}f(X_{ij})(\theta_j^T e_{i} + b_i + b_j' - \log(X_{ij}))$$

  - This ensures the embedding vectors capture the co-occurrence probabilities of words effectively.  

**Specifics of the Algorithm**
- **Handling Zero Co-occurrences:** When $X_{ij} = 0$, the term is ignored to avoid undefined values for $\log(0)$.  
- **Weighting Function:**  
  - To balance the influence of frequent (e.g., "the," "is") and rare (e.g., "durian") words, GloVe uses a weighting function $f(X_{ij})$.  
  - This function ensures that neither frequent nor infrequent words dominate the learning process.  

- **Symmetric Roles of $\theta$ and $e$:**  
  - Both $\theta$ and $e$ play symmetric roles in the model.  
  - After training, the final embedding for a word is computed as the average of $\theta$ and $e$ vectors, i.e., $e_{w}^{final} = \dfrac{\theta_w + e_w}{2}$  

**Interpretability of Embeddings**

- While embeddings are expected to capture interpretable features like gender, royalty, or age, the dimensions are often not directly meaningful.  
- The learning algorithm distributes concepts across dimensions in a way that optimizes co-occurrence predictions, not human interpretation.  

- Despite the lack of interpretability in individual dimensions, embeddings capture relationships.  
- For instance, analogies like "king - man + woman = queen" can be visualized as a parallelogram in the embedding space.  

**In summary,**
GloVe builds on earlier models like Word2Vec and skip-gram while simplifying the learning process. By leveraging co-occurrence statistics, it efficiently learns high-quality embeddings that capture word relationships, even though individual dimensions may not be interpretable. The power of GloVe lies in its ability to represent complex relationships, making it a cornerstone in natural language processing.  


### Application using Word Embeddings

#### Sentiment Classification

**Definition**: Determines whether a text expresses positive or negative sentiment, a key task in Natural Language Processing (NLP).
**Applications:** Analyzes feedback on products, services, restaurant reviews, or social media comments to gauge sentiment, identify issues, or monitor trends.

**Challenges and Solutions:**
- *Challenges:* Limited availability of labeled training data.
- *Solution:* Word embeddings enable efficient sentiment classification, even with small datasets, by leveraging large text corpora to learn meaningful representations, including rare words.

![sentiment-model](images/sentiment-model-simple.png)
- Convert sentence words into one-hot vectors using a dictionary (e.g., 10,000 words).
- Extract word embeddings using a pre-trained embedding matrix.
- Average or sum word embeddings for the sentence.
- Use a softmax classifier to predict sentiment.
- This approach efficiently handles both short and long reviews, summarizing the overall meaning of a text.


**Limitations of Simple Models:**

- Ignore word order, leading to potential misinterpretations (e.g., “Completely lacking in good taste” may be classified positively due to the presence of "good").

**Advanced Model: RNN for Sentiment Classification:**

- Instead of averaging word embeddings, a Recurrent Neural Network (RNN) processes words sequentially.
- Implements a many-to-one RNN architecture that captures the word sequence, handling nuances like “not good” versus “good.”
- Provides a deeper understanding of sentiment by considering context.
![advance-sentiment-model](images/sentiment-model-rnn.png)

**Advantages of RNN Models:**
- Accurately interprets word order and context.
- Generalizes better to new words or phrasing.
- Effectively uses words not present in the labeled dataset but included in the embedding corpus for classification.

#### Debiasing Word Embeddings