# Vast CSV Analyst

Self-hosted conversational analytics for CSV data, powered by Qwen running on
a Vast.ai GPU.

Place a standard CSV in `/workspace`, select it in the Streamlit interface, and
ask questions in plain English. PandasAI generates and executes the analysis,
while Qwen is served through vLLM on the same Vast instance rather than through
a hosted LLM API.

## What it does

- Answers standalone questions and contextual follow-ups.
- Returns written answers, numbers, tables, generated Python, and PNG charts.
- Provides **Fast** mode for straightforward questions and **Deep** mode for
  analysis that benefits from additional reasoning.
- Saves generated charts under `/workspace/charts` and makes them downloadable
  from the chat.

## Architecture

```text
Request: CSV + question -> Streamlit -> PandasAI -> LiteLLM -> Qwen/vLLM on Vast
Result:  Streamlit <- text, table, code, or chart <- PandasAI executes Qwen-generated analysis
```

[`app.py`](app.py) contains the application. The
[`pandas_ai_demo.ipynb`](pandas_ai_demo.ipynb) notebook prepares the compatible
Python environment and launches Streamlit from Jupyter.

The project is designed as a forkable reference implementation, with model
configuration, data loading, analysis, and response rendering kept as separate
concerns.

## Tested configuration

| Component     | Configuration                         |
| ------------- | ------------------------------------- |
| GPU           | 1x NVIDIA RTX PRO 6000 WS, 96 GB VRAM |
| Disk          | 120 GB                                |
| Container     | `vastai/vllm:v0.27.1-cuda-13.0`       |
| Model         | `Qwen/Qwen3.8-27B`                    |
| Model context | 32,000 tokens                         |
| vLLM endpoint | `http://127.0.0.1:18000/v1`           |

This is the verified deployment used for the project, not a claimed minimum.
The first startup can take several minutes while vLLM downloads and compiles
the model.

## Run on Vast

1. Create an instance from Vast's vLLM template with the model above, a
   high-memory NVIDIA GPU, and 120 GB of disk.
2. Wait for the model endpoint to respond:

   ```bash
   curl http://127.0.0.1:18000/v1/models
   ```

3. Put `app.py`, `pandas_ai_demo.ipynb`, and one or more CSV files directly in
   `/workspace` using Jupyter's file browser.
4. Open the notebook and run its cells in order. It creates a Python 3.11
   environment for PandasAI and explains when to refresh Jupyter and switch
   kernels.
5. Leave the final cell running, then create a Vast tunnel to
   `http://localhost:8501` and open the resulting URL.
6. Select a CSV and mode in the sidebar, then start chatting.

## Notes

The video walkthrough uses a locally prepared subset of public
[OpenIPF](https://www.openipf.org/) competition data. The application itself is
dataset-agnostic and accepts any standard CSV placed directly in `/workspace`.

Conversation state is kept within the current Streamlit session. PandasAI
executes model-generated Python in the application process, so this reference
setup is intended for trusted users and data.
