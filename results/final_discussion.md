# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

-   Number of products used: 13332

-   Changes to sampling strategy: none.

    We used the original strategy with DuckDB to limit the number of rows, so the whole data does not materialise in memory. Hence, the raw size is only a few MB (\~3-9MB).

### LLM Experiment

-   **Models compared (name, family, size)**

    Our original model was `Llama-3.2-3B-Instruct` from from the Llama 3.2 (Meta) family with size 3B parameters. We are comparing it with a second model `Qwen3-8B` from the Qwen family with size 8B parameters.

-   **Results and discussions**

    -   **Prompt used:**

        "You are a helpful Amazon shopping assistant. Answer the question using ONLY the following context (real product reviews + metadata). Always cite the product ASIN when possible.

        Customer Reviews: {context}

        Question: {query}

        Answer based on the reviews above:"

    -   **Results**

        -   **Query 1: Bar Soap**
            -   LLM 1: `Llama-3.2-3B-Instruct`

                ![](images/clipboard-3251966619.png)

            -   LLM 2: `Qwen3-8B`

                ![](images/clipboard-1636650054.png)

                For searching bar soap product, LLM 2 performed better by returning only relevant bar soap recommendations and acknowledging that the soap pouch is not a bar soap. However, LLM 2 is not confident with the hygiene soap bar which would be actually useful for user's intent. LLM 1, on the other hand, hallucinated by including a soap pouch as its first recommendation, which is an accessory rather than the product itself.
        -   **Query 2: Hairspray**
            -   LLM 1: `Llama-3.2-3B-Instruct`

                ![](images/clipboard-3061211910.png)

            -   LLM 2: `Qwen3-8B`

                ![](images/clipboard-504478598.png)

                Both models recommended the same primary product (Frizz Ease Hair Spray). The performance for the two LLMs are equal for this query.
        -   **Query 3: Humidifier**
            -   LLM 1: `Llama-3.2-3B-Instruct`

                ![](images/clipboard-672959851.png)

            -   LLM 2: `Qwen3-8B`

                ![](images/clipboard-1188413142.png)

                Both models returned the same diffuser recommendations which is the Aromyst Ultrasonic Glass Diffuser which is very relevant to user's intent. However, LLM 2 keep listing irrelevant essential oil products and notably acknowledged their irrelevance, suggesting a tendency to over-generate rather than staying focused on the query.
        -   **Query 4: Lamp**
            -   LLM 1: `Llama-3.2-3B-Instruct`

                ![](images/clipboard-2806423678.png)

            -   LLM 2: `Qwen3-8B`

                ![](images/clipboard-3254034001.png)

                Both models recommend a daylight lamp that reviewers confirmed works as a daylight alarm, which is directly matching the user's intent of finding a lamp that aids in waking up. LLM 2 recommend further products that are not actual lamp, like alarm clock, and suggesting more duplicates.
        -   **Query 5: Sunscreen**
            -   LLM 1: `Llama-3.2-3B-Instruct`

                ![](images/clipboard-2767093812.png)

            -   LLM 2: `Qwen3-8B`

                ![](images/clipboard-3089058699.png)

                Both LLMs suggested Mountain Falls Sunscreen which is an accurate sunscreen product to recommend for top choice. Nonetheless, both models failed to find a sunscreen suitable for babies, suggesting a limitation in the retrieved context rather than the models themselves. Again, the additional products LLM 2 suggesting are not relevant/accurate anymore for our query.

-   **Which model you chose and why**

    We selected LLM 1 (`Llama-3.2-3B-Instruct`) as our default model. Across the five queries, Llama demonstrated better precision and more relevant enough recommendations overall without over-generating. LLM 1 is better at keeping the result concise compared to LLM 2.

## Step 2: Additional Feature (Tool Integration)

### What You Implemented

-   **Description of the feature**

    Added web search functionality using Tavily. Both web search and RAG are exposed as tools to the agent, who is instructed to always use the RAG tool and use the web search tool when the user asks for current information.

-   Key results or examples

-   Show 3 example queries where the tool was used

-   Explain whether it improved the results

## Step 3: Improve Documentation and Code Quality

### Documentation Update

-   Summary of `README` improvements

### Code Quality Changes

-   Moved file paths into config file
-   API key already not in code
-   Added docstrings to new functions (to verify)
-   Updated environment file (to verify)
-   .gitignore already updated

## Step 4: Cloud Deployment Plan

1.  Data Storage: Where will you store the following?

    -   raw data

    -   processed data

    -   vector index

    -   BM25 index

2.  Compute

    -   Where will your app run?

    -   How will you handle multiple users (concurrency)?

    -   How will you handle LLM inference (API vs hosted model)?

3.  Streaming/Updates

    -   How will you incorporate new products in production?

    -   How will your pipeline stay up to date?
