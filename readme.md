<!--
 * @Author: Zerui Han <hanzr.nju@outlook.com>
 * @Date: 2025-06-24 11:29:48
 * @Description: 
 * @FilePath: /arxiv-summary/readme.md
 * @LastEditTime: 2025-07-08 18:13:31
-->
# Arxiv Summarizer with OpenAI and Feishu Webhook

[简体中文](readme.zh.md)

This project automatically summarizes recent Arxiv papers in a specified subject category using the OpenAI API and sends the summary to a Feishu (Lark) webhook. It's designed to run regularly using GitHub Actions.

## Prerequisites

Before you can use this project, you'll need the following:

1.  **OpenAI API Key:** You'll need an API key from OpenAI to use their language models.  Create an account and obtain your API key from [OpenAI](https://platform.openai.com/).

2.  **Feishu (Lark) Webhook URL:** You'll need to create an incoming webhook in your Feishu group to receive the summaries.  Refer to the Feishu documentation on how to set up a webhook: [Feishu Incoming Webhooks](https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot?lang=zh-CN).  **Important:** Make sure your webhook has the necessary permissions in Feishu to post messages to the desired group.

## How to Run

You can run this project in two ways: automatically via GitHub Actions or manually on your local machine.

### Option 1: Running with GitHub Actions

This is the recommended method for automated daily summaries.

1.  **Fork the Repository:** Click the "Fork" button in the top-right corner of this GitHub repository to create a copy of the project in your own GitHub account.

2.  **Configure GitHub Secrets:**

    *   Go to your forked repository's page on GitHub.
    *   Click on the "Settings" tab.
    *   In the left sidebar, click on "Security", then "Secrets and variables", and finally "Actions."
    *   Add the following secrets:
        *   **`OPENAI_API_KEY`:** Your OpenAI API key.
        *   **`OPENAI_BASE_URL`:** (Optional) The base URL for the API endpoint. **Important:** If you are using a proxy, a different endpoint for OpenAI, or an OpenAI-compatible API provider (e.g., [ModelScope](https://www.modelscope.cn/docs/model-service/API-Inference/intro)), you must set this variable. If you're using the standard OpenAI API, leave this blank.
        *   **`WEBHOOK_URL`:** Your Feishu webhook URL.

3.  **Configure GitHub Variables:**

    *   In the same "Settings -> Security -> Secrets and variables -> Actions -> Variables" section, add the following variables:
        *   **`OPENAI_MODEL_NAME`:** The name of the OpenAI model you want to use for summarization (e.g., `gpt-3.5-turbo`, `gpt-4`). Refer to the OpenAI documentation for available models.
        *   **`SUMMARY_LANGUAGE`:** The language in which you want the summary to be generated (e.g., `English`, `Chinese`).

4.  **Configure and Enable the Workflow:**

    *   The workflow file (`.github/workflows/eess.AS.yaml`) is already set up to run the summarizer every day at 00:02 UTC (08:02 Beijing Time). You can edit the `cron` expression in the file to change the schedule.
    *   **Note on Scheduling:** Please be aware that there can be a significant latency for scheduled GitHub Actions. For example, a workflow scheduled for 08:02 Beijing Time might not execute until around 09:50. Additionally, Arxiv typically refreshes its daily papers around 09:00 Beijing Time. Therefore, it is recommended not to set the schedule too early to ensure you are summarizing the latest papers.
    *   GitHub Actions should be enabled by default. If not, go to the "Actions" tab in your repository and enable them.
    *   The summarizer will now run automatically. You can also trigger a manual run by going to the "Actions" tab, selecting the "Arxiv Summarizer" workflow, and clicking "Run workflow".

### Option 2: Running Locally

Follow these steps to run the summarizer on your local machine.

1.  **Create a Virtual Environment:** It's recommended to use a virtual environment to manage dependencies.

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies:** Install the required Python packages.

    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Environment Variables:** Create a file named `.env` in the root of the project and add the following content. The script will automatically load these variables.

    ```.env
    OPENAI_API_KEY="your_openai_api_key"
    WEBHOOK_URL="your_feishu_webhook_url"
    OPENAI_MODEL_NAME="gpt-3.5-turbo"
    SUMMARY_LANGUAGE="English"
    # Optional: If you are using a proxy or a different endpoint
    # OPENAI_BASE_URL="your_openai_base_url"
    ```

    *Alternatively, you can still use `export` to set them in your shell, but a `.env` file is recommended for ease of use.*

4.  **Run the Script:** Execute the script with your desired arguments.

    ```bash
    python arxiv_summarizer.py eess.AS cs.AI --user_interest "speech processing, audio synthesis" --filter_level "mid"
    ```
    You can provide multiple categories separated by spaces.

5.  **Schedule with Cron (macOS/Linux):** To run the script automatically on a schedule, you can use `cron`.

    *   Open your crontab for editing:
        ```bash
        crontab -e
        ```
    *   Add a new line to run the script at your desired time. For example, to run it every day at 9:01 AM:
        ```cron
        1 9 * * * /path/to/your/project/venv/bin/python /path/to/your/project/arxiv_summarizer.py YOUR_CATEGORY
        ```
        *Remember to replace `/path/to/your/project` with the actual path to this project directory.*

    *   **Note for Windows Users:** Windows does not have `cron`. You can use the built-in **Task Scheduler** to achieve the same result. You will need to create a new task that runs the Python script at your desired interval.



## Customization

*   **Subject Category, User Interest, and Filtering:** The script currently summarizes papers from the `eess.AS` category (Audio and Speech Processing). You can customize the behavior by modifying the arguments passed to the `arxiv_summarizer.py` script in the `.github/workflows/eess.AS.yml` file:

    ```yaml
        run: python arxiv_summarizer.py "eess.AS" "cs.AI" --user_interest "your, interests" --filter_level "mid"
    ```

    *   Replace `"eess.AS" "cs.AI"` with your desired Arxiv subject category codes (e.g., `cs.AI`, `math.ST`). You can provide one or more categories separated by spaces. Refer to the Arxiv documentation for a list of available categories.
    *   Use `--user_interest` to specify your specific areas of interest (e.g., `"machine learning, NLP"`). If provided, papers will be scored for relevance and sorted. If omitted, all papers will have relevance 0.
    *   Use `--filter_level` to filter papers based on relevance. Options are `low` (score >=0), `mid` (score >=1), `high` (score >=2), or `none` (no filtering). If a filter level is set, only papers with relevance higher than or equal to the specified level will be summarized. This can help save tokens by avoiding summarization of less relevant papers.

## Important Notes

*   **Cost:** Using the OpenAI API can incur costs.  Monitor your OpenAI API usage and set up billing alerts to avoid unexpected charges.

*   **Rate Limiting:** The Arxiv website and the OpenAI API may have rate limits. The script should handle rate limiting gracefully, but you may need to adjust the request frequency if you encounter errors.

*   **Error Handling:**  The script includes basic error handling, but you may need to add more robust error handling for production use.

*   **Security:** Do not hardcode your API keys or webhook URLs in the code. Always use GitHub Secrets for sensitive information.

*   **Arxiv Usage:** Be respectful of Arxiv's usage policies and avoid excessive scraping.