# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling
- Number of products used: 13332
- Changes to sampling strategy: none. Used original strategy of DuckDB to limit the number of rows so the whole data does not materialise in memory. The raw size is a few MB.

### LLM Experiment
- Models compared (name, family, size)
- Results and discussions
    - Prompt used (copy it here)
    - Results
- Which model you chose and why

## Step 2: Additional Feature (Tool Integration)

### What You Implemented

- Added web search functionality using Tavily. Both web search and RAG are exposed as tools to the agent, who is instructed to always use the RAG tool and use the web search tool when the user asks for current information.
- Key results or examples
- Show 3 example queries where the tool was used
- Explain whether it improved the results
  
## Step 3: Improve Documentation and Code Quality

### Documentation Update
- Summary of `README` improvements

### Code Quality Changes
- Moved file paths into config file
- API key already not in code
- Added docstrings to new functions (to verify)
- Updated environment file (to verify)
- .gitignore already updated

## Step 4: Cloud Deployment Plan
(See Step 4 above for required subsections)