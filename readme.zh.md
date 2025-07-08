<!--
 * @Author: Zerui Han <hanzr.nju@outlook.com>
 * @Date: 2025-06-26 10:25:19
 * @Description: 
 * @FilePath: /arxiv-summary/readme.zh.md
 * @LastEditTime: 2025-07-08 18:16:13
 -->
 # Arxiv 论文摘要工具 (OpenAI + 飞书)

[English](readme.md)

本项目自动抓取并通过 OpenAI API 总结 Arxiv 上指定领域的最新论文，并将摘要推送到飞书（Lark）群聊。整个流程通过 GitHub Actions 定时运行，方便你每日跟进最新科研动态。

## 准备工作

开始之前，请确保你已准备好以下各项：

1.  **OpenAI API Key：** 你需要一个 OpenAI API Key 来调用模型。前往 [OpenAI 官网](https://platform.openai.com/) 注册并获取 API Key。

2.  **飞书 (Lark) Webhook URL：** 一个飞书群聊的 Webhook URL，用于接收论文摘要。具体设置方法请参考飞书官方文档：[获取自定义机器人Webhook](https://open.feishu.cn/document/client-docs/bot-v3/add-custom-bot?lang=zh-CN)。**注意：** 请确保机器人有在目标群聊中发送消息的权限。

## 如何运行

你可以通过两种方式运行此项目：使用 GitHub Actions 自动运行，或在你的本地机器上手动运行。

### 方式一：使用 GitHub Actions 运行

这是推荐的自动每日摘要方法。

1.  **Fork 本仓库：** 点击页面右上角的 `Fork` 按钮，将此项目复制到自己的 GitHub 账号下。

2.  **配置 GitHub Secrets：**
    *   进入你 Fork 后的仓库，找到 `Settings` > `Secrets and variables` > `Actions`。
    *   点击 `New repository secret` 添加以下三项：
        *   `OPENAI_API_KEY`：OpenAI API 密钥。
        *   `OPENAI_BASE_URL`：(可选) API 代理地址。如果你使用了代理，或者其他兼容 OpenAI 接口的 API 服务（如 [ModelScope](https://www.modelscope.cn/docs/model-service/API-Inference/intro)），则需要配置此项。如果直连 OpenAI 官方接口，请留空。
        *   `WEBHOOK_URL`：飞书 Webhook URL。

3.  **配置 GitHub Variables：**
    *   在同一页面 (`Settings` > `Secrets and variables` > `Actions`)，切换到 `Variables` 标签页，添加以下变量：
        *   `OPENAI_MODEL_NAME`：用于摘要的 OpenAI 模型名称（例如 `gpt-3.5-turbo`, `gpt-4`）。
        *   `SUMMARY_LANGUAGE`：摘要的目标语言（例如 `English`, `Chinese`）。

4.  **配置并启用工作流：**
    *   工作流文件 (`.github/workflows/eess.AS.yaml`) 默认设置为每天 UTC 00:02（即北京时间 08:02）运行。你可以通过修改文件中的 `cron` 表达式来调整运行时间。
    *   **关于时间的建议：** GitHub Actions 的定时任务存在延迟，一个 08:02 的任务可能到 09:50 才实际执行。同时，Arxiv 通常在北京时间上午 9 点左右更新论文。因此，建议不要将 `cron` 任务设置得太早，以免错过当天的最新内容。
    *   Fork 后的仓库默认会启用 Actions。如果未启用，请手动前往仓库的 `Actions` 标签页开启。
    *   现在，摘要工具将自动运行。你也可以随时进入仓库的 `Actions` 标签页，选择 `Arxiv Summarizer` 工作流，手动触发一次运行。

### 方式二：在本地运行

按照以下步骤在你的本地计算机上运行此摘要工具。

1.  **创建虚拟环境：** 推荐使用虚拟环境来管理依赖。

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **安装依赖：** 安装所需的 Python 包。

    ```bash
    pip install -r requirements.txt
    ```

3.  **设置环境变量：** 在项目根目录创建一个名为 `.env` 的文件，并添加以下内容。脚本会自动加载这些变量。

    ```.env
    OPENAI_API_KEY="你的_openai_api_key"
    WEBHOOK_URL="你的_feishu_webhook_url"
    OPENAI_MODEL_NAME="gpt-3.5-turbo"
    SUMMARY_LANGUAGE="Chinese"
    # 可选：如果你使用代理或不同的端点
    # OPENAI_BASE_URL="你的_openai_base_url"
    ```

    *当然，你仍然可以使用 `export` 命令在终端中设置这些变量，但推荐使用 `.env` 文件以便管理。*

4.  **运行脚本：** 使用你想要的参数执行脚本。

    ```bash
    python arxiv_summarizer.py eess.AS cs.AI --user_interest "语音处理, 音频合成" --filter_level "mid"
    ```
    你可以提供多个领域，用空格分隔。

5.  **使用 Cron (macOS/Linux) 定时运行：** 要让脚本按计划自动运行，你可以使用 `cron`。

    *   打开你的 crontab 进行编辑：
        ```bash
        crontab -e
        ```
    *   添加新的一行，在你希望的时间运行脚本。例如，每天早上 9:01 运行：
        ```cron
        1 9 * * * /path/to/your/project/venv/bin/python /path/to/your/project/arxiv_summarizer.py 你想关注的领域
        ```
        *请记得将 `/path/to/your/project` 替换为本项目的实际路径。*

    *   **Windows 用户注意：** Windows 系统沒有 `cron`。你可以使用內置的 **任务计划程序 (Task Scheduler)** 來实现相同的效果。你需要创建一个新任务，在指定的时间间隔运行 Python 脚本。



## 自定义

脚本默认总结 `eess.AS` (音频与语音处理) 领域的论文。你可以通过修改 `.github/workflows/eess.AS.yml` 文件中的启动命令，来自定义关注的领域、兴趣点和过滤强度：

```yaml
      run: python arxiv_summarizer.py "eess.AS" "cs.AI" --user_interest "你的兴趣点" --filter_level "mid"
```

*   将 `"eess.AS" "cs.AI"` 替换为感兴趣的 Arxiv 领域代码（例如 `cs.AI`, `math.ST`）。你可以提供一个或多个领域，用空格分隔。你可以在 Arxiv 官网找到所有领域的代码。
*   `--user_interest`：指定你的具体研究兴趣，用逗号分隔（例如 `"机器学习, 自然语言处理"`）。如果提供了此参数，脚本会根据相关性对论文进行打分和排序；否则，所有论文相关性分数为 0。
*   `--filter_level`：根据相关性分数过滤论文。可选值为 `low` (分数>=0), `mid` (分数>=1), `high` (分数>=2) 或 `none` (不过滤)。设置过滤等级后，只有高于或等于该等级的论文才会被总结。这可以有效节省无关论文的 token 开销。

## 注意事项

*   **成本：** 使用 OpenAI API 会产生费用。请密切关注你的用量，并设置账单提醒，避免产生意外开销。

*   **速率限制：** Arxiv 和 OpenAI API 都可能存在速率限制。脚本已包含基本的重试逻辑，但如果频繁出错，你可能需要调整运行频率或优化代码。

*   **错误处理：** 脚本包含基础的错误处理，但对于生产环境或更复杂的应用场景，你可能需要自行增强其健壮性。

*   **安全性：** **切勿**在代码中硬编码你的 API Key 或 Webhook URL。务必使用 GitHub Secrets 进行管理。

*   **Arxiv 使用：** 请遵守 Arxiv 的使用政策，避免过于频繁地请求，给服务器造成不必要的负担。