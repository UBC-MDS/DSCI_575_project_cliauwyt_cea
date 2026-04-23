# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

-   Number of products used: 13332

-   Changes to sampling strategy: none.

    We used the original strategy with DuckDB to limit the number of rows, so the whole data does not materialise in memory. Hence, the raw size is only a few MB (\~3-9MB).

### LLM Experiment

-   **Models compared (name, family, size)**

    Our original model was `Llama-3-8B-Instruct` from from the Llama 3 (Meta) family with size 8B parameters. We are comparing it with a second model `Qwen3-8B` from the Qwen family with size 8B parameters.

-   **Results and discussions**

    -   **Prompt used:**

        "You are a helpful Amazon shopping assistant. Answer the question using ONLY the following context (real product reviews + metadata). Always cite the product ASIN when possible.

        Customer Reviews: {context}

        Question: {query}

        Answer based on the reviews above:"

    -   **Results**

        -   **Query 1: Bar Soap**
            -   LLM 1: `Llama-3-8B-Instruct`

                ![](../img/M2/bar_soap_rag.png)

            -   LLM 2: `Qwen3-8B`

                ![](../img/Final/barsoap_rag2.png)

                For searching bar soap product, LLM 2 performed better by returning only relevant bar soap recommendations and acknowledging that the soap pouch is not a bar soap. However, LLM 2 was not confident with the hygiene soap bar which would be actually useful for user's intent. LLM 1, on the other hand, included a soap pouch as its first recommendation, which is an accessory rather than the product itself.
        -   **Query 2: Hairspray**
            -   LLM 1: `Llama-3-8B-Instruct`

                ![](../img/M2/hairspray_rag.png)

            -   LLM 2: `Qwen3-8B`

                ![](../img/Final/hairspray_rag2.png)

                Both models recommended the same primary product (Frizz Ease Hair Spray). The performance for the two LLMs are equal for this query.
        -   **Query 3: Humidifier**
            -   LLM 1: `Llama-3-8B-Instruct`

                ![](../img/Final/humidifier_rag1.png)

            -   LLM 2: `Qwen3-8B`

                ![](../img/Final/humidifier_rag2.png)

                Both models returned the same diffuser recommendations which is the Aromyst Ultrasonic Glass Diffuser which is very relevant to user's intent. However, LLM 2 kept listing irrelevant essential oil products and notably acknowledged their irrelevance, suggesting a tendency to over-generate rather than staying focused on the query.
        -   **Query 4: Lamp**
            -   LLM 1: `Llama-3-8B-Instruct`

                ![](../img/M2/lamp_rag.png)

            -   LLM 2: `Qwen3-8B`

                ![](../img/Final/lamp_rag2.png)

                Both models recommend a daylight lamp that reviewers confirmed works as a daylight alarm, which is directly matching the user's intent of finding a lamp that aids in waking up. LLM 2 recommended further products that are not actual lamps, like alarm clocks, and suggesting more duplicates.
        -   **Query 5: Sunscreen**
            -   LLM 1: `Llama-3-8B-Instruct`

                ![](../img/Final/sunscreen_rag1.png)

            -   LLM 2: `Qwen3-8B`

                ![](../img/Final/sunscreen_rag2.png)

                Both LLMs suggested Mountain Falls Sunscreen which is an accurate sunscreen product to recommend for top choice. Nonetheless, both models failed to find a sunscreen suitable for babies, suggesting a limitation in the retrieved context rather than the models themselves. Again, the additional products LLM 2 suggesting are not relevant/accurate anymore for our query.

-   **Which model we chose and why**

    We selected LLM 1 (`Llama-3-8B-Instruct`) as our default model. Across the five queries, Llama demonstrated better precision and more relevant enough recommendations overall without over-generating. LLM 1 is better at keeping the result concise compared to LLM 2.

## Step 2: Additional Feature (Tool Integration)

### What You Implemented

-   **Description of the feature**

    To extend the pipeline beyond static document retrieval, web search functionality was integrated using Tavily. Both the RAG retriever and the web search tool are exposed as tools to the agent. The agent is instructed to always prioritize the RAG tool for product-related queries, and use the web search tool when the user requests current or real-time information. This allows the pipeline to ground its responses in both curated product reviews and up-to-date external information.

-   **Three Key results or examples**

    1.  What soap is trending right now

    ![](../img/Final/trend_soap_fail_tool.png)

    It seems that the agent failed to call the web-search tool instead of RAG tool for retrieving "what soap is trending right now".

    2.  What soap is famous right now

    ![](../img/Final/famous_soap_rag.png)

    Here is an example of a regular RAG tool query result to compare with Agent Mode.

    ![](../img/Final/famous_soap_tool.png)

    For this query, the agent succeeded to call the web-search tool to retrieve some real-time information about current popular soaps. Interestingly, the agent also called the RAG tool to retrieve some products from our data set, which is similar to using RAG mode (except the Boyer Lye product).

    3.  Is there any recalled sunscreen in 2025

    ![](../img/Final/recalled_sunscreen_tool.png)

    The agent was able to explain if there were any recalled sunscreens in 2025 from recent articles.

-   **Explain whether it improved the results**

    The web-search tool improves the results by adding real-time information which would be out of the scope of our original data set. As a result, the response becomes less static and more flexible since we are not relying on a static data set. However, for standard product recommendation queries, the web search tool adds little benefit over pure RAG since both would recommend the same products from the data set. Additionally, the agent's ability to select the right tool for the right query is critical to overall performance.

## Step 3: Improve Documentation and Code Quality

### Documentation Update

-   Added description for tool implementation
-   Added set up information on how to acquire Tavily API key and access the Agent mode for our App
-   Added a demo video as a usage example in the end of the `README` file

### Code Quality Changes

-   Moved file paths into config file
-   API key already not in code
-   Added docstrings to new functions
-   Updated environment file
-   .gitignore already updated

## Step 4: Cloud Deployment Plan

We would deploy our Amazon Product Query Assistant pipeline on AWS.

1.  Data Storage:

    All data (raw + processed) and indexes (BM25 + Semantic + vector) will be stored in Amazon S3 due to low cost, scalability, and ease of access from other AWS services like EC2 and Lambda. All indexes will be serialized (e.g. as `.pkl` or `.faiss` files) before storing into S3. When the app starts, the indexes are downloaded from S3 and loaded into memory. This ensures indexes persist between sessions without needing to be rebuilt every time the app restarts.

2.  Compute

    The app will run on an AWS EC2 instance. The app's dependencies will be managed using a Python virtual environment, we use `venv` + `pip` with `requirement.txt` file to ensure a consistent runtime environment.

    Using AWS Auto Scaling can help to handle concurrency by automatically spin up additional EC2 instances when traffic increases. An AWS Elastic Load Balancer (ELB) would distribute incoming requests across instances to ensure multiple users can query the system simultaneously without compromising the App performance.

    For LLM inference, we will continue using an external inference API (HuggingFace Inference API via Novita) rather than self-hosting a model. Therefore, setting API key from HuggingFace in `.env` variables is required to run the App. Using API avoids the high cost of GPU EC2 instances which are expensive to run continuously.

3.  Streaming/Updates

    For incorporating new reviews, the current `download_data.ipynb` notebook will be converted to a proper script with the row limit removed. New reviews will be fetched by connecting to the Amazon product and reviews data set API using DuckDB, which supports incremental/streaming queries. Only newly added reviews will be added to the existing data set in S3 using incremental pipeline strategy with `append`, avoiding a full re-download each time. Then, all indexes can be rebuilt to stay updated with the new added data.

    Using AWS Lambda service will help the pipeline stay up to date by running on a scheduled basis (e.g. daily or weekly) via AWS EventBridge. The pipeline will run the following steps automatically:

    1.  Connect to the Amazon dataset API using DuckDB and download only new data incrementally
    2.  Preprocess the data and rebuild the indexes (BM25 + Semantic + vectorstore)
    3.  Upload the updated indexes back to S3

    Since Lambda is event-driven and serverless, it is cost-effective for periodic update jobs like this. We save the cost compared to keeping a server running continuously. This ensures the recommendation pipeline stays up to date with new products and reviews without manual intervention.
