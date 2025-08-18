import datetime
import io
import json
import logging
import os
import sys
import tarfile

import arxiv
import openai
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv


class ArxivSummarizer:
    """
    A class to automatically summarize arXiv papers using LLMs.
    """

    def __init__(self):
        """
        Initializes the ArxivSummarizer class.
        Loads environment variables, sets up logging, and initializes the OpenAI client.
        """
        self._setup_logging()  # Initialize logging
        self._load_environment_variables()
        self.client = openai.OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            max_retries=5,  # Enable retries
        )

    def _setup_logging(self):
        """Configures logging."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],  # Output to console
        )

    def _load_environment_variables(self):
        """Loads environment variables from .env file."""
        load_dotenv()  # Load .env if it exists

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        self.openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")  # Default model
        self.summary_language = os.getenv("SUMMARY_LANGUAGE", "English")  # Default language
        self.webhook_url = os.getenv("WEBHOOK_URL")

        if not self.openai_api_key:
            logging.error("OPENAI_API_KEY not found in environment variables.")
            raise ValueError("OPENAI_API_KEY is required.")

    def get_author_affiliations_from_tex(self, paper_id: str) -> str | None:
        """
        Fetches and extracts author affiliations from the TeX source of an arXiv paper.
        """
        try:
            url = f"https://arxiv.org/src/{paper_id}"
            logging.info(f"Fetching TeX source from: {url}")
            response = requests.get(url)
            response.raise_for_status()

            with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
                tex_files = [m for m in tar.getmembers() if m.name.endswith(".tex")]
                if not tex_files:
                    logging.warning(f"No .tex files found in the archive for paper {paper_id}.")
                    return None

                for member in tex_files:
                    tex_content = tar.extractfile(member).read().decode("utf-8", errors="ignore")
                    if r"\documentclass" in tex_content:
                        lines = tex_content.splitlines()
                        # Filter out lines containing LaTeX comments
                        lines = [line for line in lines if not line.startswith("% ")]
                        start_index = 0
                        end_index = min(len(lines), 150)
                        context_lines = lines[start_index:end_index]
                        context_lines = self._filter_latex_lines(context_lines)
                        joined_context = "\n".join(context_lines)

                        prompt = f"""The following lines are extracted from a .tex file.
                        Your task is to identify and extract the author affiliations.
                        Tex content:
                        {joined_context}
                        Output Format:
                        - Only extract the top-level affiliation (e.g., University name, Company name, Research Institute name), avoiding overly specific details like departments, schools, or specific addresses.
                        - If there are affiliations, list them without numbering, separated by semicolons (;).
                        - If the number of unique top-level affiliations is more than three, list the first three, followed by \"etc.\" to indicate the rest.
                        - Respond in {self.summary_language}.
                        - If no affiliations are found, respond with 'None'.
                        - **Do not respond with anything other than the affiliations!**
                        Examples of desired output (assuming English, do not include the double quotes):
                        - \"University of Example; Tech Innovations Inc.\"
                        - \"University A; Company B; Research Institute C; etc.\"
                        - \"None\"
                        Examples of what to avoid (too detailed):
                        - \"Department of Physics, University of Example\" -> should be \"University of Example\"
                        - \"AI Lab, Company C, Country P\" -> should be \"Company C\"
                        - \"School of Computer Science, University Z, City X\" -> should be \"University Z\"
                        """

                        completion = self.client.chat.completions.create(
                            model=self.openai_model_name,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0,
                        )
                        affiliations = completion.choices[0].message.content.strip()
                        if affiliations.lower().startswith("none"):
                            return None
                        return affiliations
            return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error fetching TeX source for paper {paper_id}: {e}")
            return None
        except tarfile.TarError as e:
            logging.error(f"Error extracting TeX source for paper {paper_id}: {e}")
            return None
        except Exception as e:
            logging.error(f"Error processing TeX source for paper {paper_id}: {e}")
            return None

    @staticmethod
    def _filter_latex_lines(lines_list: list[str]) -> list[str]:
        """Filter out lines that seem not include affiliation information to save tokens."""
        exclude_prefixes = [
            "\\usepackage",
            "\\documentclass",
            "\\definecolor",
            "\\setlength",
            "\\newcommand",
            "\\renewcommand",
            "\\begin",
            "\\end",
            "\\input",
            "\\include",
            "\\section",
            "\\subsection",
            "\\subsubsection",
            "\\paragraph",
            "\\subparagraph",
            "\\label",
            "\\ref",
            "\\eqref",
            "\\cite",
            "\\fontsize",
            "\\hypersetup",
            "\\footnote",
            "\\maketitle",
            "\\date",
            "\\graphicspath",
            "\\includegraphics",
            "\\url",
            "\\href",
            "\\pagestyle",
            "\\thispagestyle",
            "\\item",
            "\\caption",
            "\\figure",
            "\\table",
            "\\keywords",
            "\\abstract",
            "\\document",
        ]
        filtered_lines = []
        for line in lines_list:
            line = line.strip()
            # Skip empty lines
            if not line:
                continue
            # Check against exclusion prefixes (case-insensitive for commands)
            is_excluded = False
            for prefix in exclude_prefixes:
                # For comments, we check directly
                if line.startswith(prefix):
                    is_excluded = True
                    break
            if not is_excluded:
                filtered_lines.append(line)
        return filtered_lines

    def get_paper_links_from_arxiv_page(self, category: str) -> list:
        """
        Fetches paper links (starting with /abs/) from an arXiv page for a given category,
        excluding replacement submissions and keeping only new and cross submissions.

        Args:
            category (str): The arXiv category to fetch papers from.

        Returns:
            list: A list of paper links (with /abs/ prefix) excluding replacements.
        """
        url = f"https://arxiv.org/list/{category}/new"
        logging.info(f"Fetching paper links from: {url}")
        try:
            response = requests.get(url, timeout=60)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.content, "html.parser")

            # Find all h3 headers which indicate submission types
            links = []
            skip_until_next_header = False

            # Iterate through all elements after the first h3
            for element in soup.find_all():
                # If we find an h3 header
                if element.name == "h3":
                    header_text = element.get_text().lower()
                    # Check if it's a replacement submission header
                    if "replacement" in header_text:
                        skip_until_next_header = True
                    else:
                        skip_until_next_header = False
                    continue

                # If we're skipping until the next header, continue
                if skip_until_next_header:
                    continue

                # If we find an 'a' tag with href starting with '/abs/'
                if element.name == "a" and element.get("href", "").startswith("/abs/"):
                    links.append(element["href"])

            logging.info(
                f"Found {len(links)} paper links (excluding replacements) for category {category}."
            )
            return links
        except requests.exceptions.RequestException as e:
            logging.error(f"Network error fetching arXiv page {url} for category {category}: {e}")
            return []
        except Exception as e:
            logging.error(f"Error parsing arXiv page {url} for category {category}: {e}")
            return []

    def get_all_paper_links(self, categories: str | list[str]) -> list:
        """
        Fetches paper links for one or more arXiv categories, removes duplicates, and returns a combined list.
        """
        all_links = set()
        if isinstance(categories, str):
            categories = [categories]  # Convert single category to a list

        for category in categories:
            links = self.get_paper_links_from_arxiv_page(category)
            all_links.update(links)

        logging.info(f"Found {len(all_links)} unique paper links across all categories.")
        return list(all_links)

    def get_paper_metadata(self, paper_id: str) -> dict:
        """
        Retrieves paper metadata (title, abstract) from arXiv using the arxiv library.
        The arxiv library typically fetches the latest version if a base ID is provided.
        """
        try:
            client = arxiv.Client()
            search = arxiv.Search(id_list=[paper_id])
            results = client.results(search)
            paper = next(results)  # Get the first result
            authors = [author.name for author in paper.authors]
            # Limit authors to avoid overly long strings
            if len(authors) > 3:
                authors = authors[:2] + ["et al."]
            affiliations = self.get_author_affiliations_from_tex(paper_id)
            if not affiliations:
                logging.info(f"No affiliations found for paper {paper_id}")
            return {
                "title": paper.title,
                "authors": ", ".join(authors),
                "affiliations": affiliations,
                "abstract": paper.summary,
                "url": paper.entry_id.replace("http://", "https://"),  # Ensure HTTPS
            }

        except Exception as e:
            # Log and re-raise, but allow process_arxiv_url to catch and continue
            logging.error(f"Error fetching metadata for paper ID {paper_id}: {e}")
            raise

    def summarize_paper(self, title: str, abstract: str) -> dict:
        """
        Summarizes the paper, translates its title, and extracts demo/code links using the OpenAI API.
        Returns a dictionary with translated_title, summary, demo, and code.
        """
        prompt = f"""Analyze the following research paper and return a JSON object with following fields: "translated_title", "summary", "keywords", "demo", and "code".
- "translated_title": Translate the paper's title to {self.summary_language}. Do not include any TeX formula.
- "summary": A summary of the paper in {self.summary_language}, up to 3 sentences. Do not include any TeX formula.
- "keywords": A list of keywords related to the paper in {self.summary_language}, up to 3.
- "demo": If the abstract mentions a demo page, provide the full URL. If not, this should be null.
- "code": If the abstract mentions a code repository (like a GitHub link), provide the full URL. If not, this should be null.

Title: {title}
Abstract: {abstract}

Your response must be a single JSON object.
"""
        try:
            completion = self.client.chat.completions.create(
                model=self.openai_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                response_format={"type": "json_object"},
            )
            response_content = completion.choices[0].message.content
            summary_data: dict = json.loads(response_content)

            # Ensure all keys are present, defaulting to None or empty string
            summary_data.setdefault("translated_title", title)
            summary_data.setdefault("summary", "")
            summary_data.setdefault("keywords", [])
            summary_data.setdefault("demo", None)
            summary_data.setdefault("code", None)

            return summary_data

        except json.JSONDecodeError as e:
            logging.error(
                f"Failed to decode JSON response for paper '{title}': {e}. Response was: {response_content}"
            )
            # Return a dict with default values so the process can continue
            return {
                "translated_title": title,
                "summary": "Summary could not be generated.",
                "demo": None,
                "code": None,
            }
        except openai.APIConnectionError as e:
            logging.error(f"Failed to connect to OpenAI API for summarization: {e}")
            raise
        except openai.RateLimitError as e:
            logging.error(f"OpenAI API rate limit exceeded for summarization: {e}")
            raise
        except Exception as e:
            logging.error(f"Error during summarization for paper '{title}': {e}")
            raise

    def evaluate_relevance(self, title: str, abstract: str, user_interest: str) -> int:
        """
        Evaluates the relevance of a paper to the user's interest using the OpenAI API.
        Returns 0 (low), 1 (medium), or 2 (high).
        """
        prompt = f"""Given the following research paper's title and abstract, and a (list of) user's area of interest,
        rate the relevance of the paper to the user's interest.
        Respond with only a single integer:
        0 for Low relevance to all of the user's interests or related to the 'not interested' topics,
        1 for Medium relevance to any of the user's interests,
        2 for High relevance to any of the user's interests.

        User's Interest: {user_interest}

        Paper Title: {title}
        Paper Abstract: {abstract}

        Relevance Score (0, 1, or 2):"""

        try:
            completion = self.client.chat.completions.create(
                model=self.openai_model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,  # Make it deterministic for score
                max_tokens=1,  # We only expect a single digit
            )
            score_str = completion.choices[0].message.content.strip()
            try:
                score = int(score_str)
                if score not in [0, 1, 2]:
                    logging.warning(
                        f"LLM returned an unexpected relevance score: '{score_str}'. Defaulting to 0 for paper '{title}'."
                    )
                    return 0
                return score
            except ValueError:
                logging.warning(
                    f"Could not parse relevance score (not an integer) from LLM for paper '{title}'. Response was: '{score_str}'. Defaulting to 0."
                )
                return 0
        except openai.APIConnectionError as e:
            logging.error(f"Failed to connect to OpenAI API for relevance check: {e}")
            raise  # Re-raise critical API errors
        except openai.RateLimitError as e:
            logging.error(f"OpenAI API rate limit exceeded for relevance check: {e}")
            raise  # Re-raise critical API errors
        except Exception as e:  # Catch any other unexpected errors from OpenAI client
            logging.error(
                f"An unexpected error occurred during relevance evaluation for paper '{title}': {e}"
            )
            return 0  # Default to low relevance on general error

    def process_arxiv_url(
        self,
        categories: str | list[str],
        user_interest: str | None = None,
        filter_level: str = "none",
    ) -> list[dict] | None:
        """
        Main function to orchestrate the process of fetching, summarizing, and evaluating papers.
        Returns a list of processed paper metadata dictionaries.
        """
        papers = []

        # Define relevance score mapping for filtering
        relevance_thresholds = {"low": 0, "mid": 1, "high": 2, "none": -1}  # -1 means no filtering
        if not user_interest and filter_level != "none":
            logging.warning(
                f"User interest not specified, but filter level '{filter_level}' is set. Skipping filtering."
            )
            filter_level = "none"
        min_relevance_score = relevance_thresholds.get(filter_level.lower(), -1)

        try:
            paper_links = self.get_all_paper_links(categories)

            for link in paper_links:
                paper_id = link.split("/")[-1]
                logging.info(f"Processing paper ID: {paper_id}")
                try:
                    metadata = self.get_paper_metadata(paper_id)
                    title = metadata["title"]
                    abstract = metadata["abstract"]

                    relevance_score = 0  # Default to 0
                    if user_interest:
                        relevance_score = self.evaluate_relevance(title, abstract, user_interest)
                        logging.info(f"Relevance for '{title}': {relevance_score}")

                    metadata["relevance"] = relevance_score

                    # Apply filtering based on relevance_score and filter_level
                    if min_relevance_score != -1 and relevance_score < min_relevance_score:
                        logging.info(
                            f"Paper '{title}' (ID: {paper_id}) has relevance {relevance_score}, which is below filter level '{filter_level}' ({min_relevance_score}). Skipping summarization."
                        )
                        continue  # Skip to the next paper

                    summary_data = self.summarize_paper(title, abstract)
                    metadata.update(summary_data)

                    papers.append(metadata)
                except (openai.APIConnectionError, openai.RateLimitError) as e:
                    # Log a warning and skip to the next paper if retries fail.
                    logging.warning(
                        f"OpenAI API error for paper ID {paper_id} after retries: {e}. Skipping paper."
                    )
                    continue
                except Exception as e:
                    # For other errors (e.g., arxiv library, parsing), just log and continue to the next paper
                    logging.error(f"Failed to process paper ID {paper_id}. Error: {e}")

            if not papers:
                logging.warning("No papers were successfully processed.")
                return None  # Indicate no papers were processed

            # Sorting is now handled in the run method.

            return papers

        except requests.exceptions.RequestException as e:
            logging.error(f"Network error fetching arXiv page or during initial processing: {e}")
            return None
        except Exception as e:
            logging.error(f"An unhandled error occurred during overall paper processing: {e}")
            return None

    def send_arxiv_data_via_webhook(self, data_list: list, workflow_name: str):
        """
        Sends Arxiv paper data in a single message to a webhook.

        Args:
            data_list: A list of dictionaries, where each dictionary represents an Arxiv paper
                    and contains keys like 'title', 'url', 'authors', 'abstract', 'summary'.
            category_with_suffix: The Arxiv category string, potentially with a batch suffix.

        Returns:
            The response object from the webhook for a successful send, or None if there's an error.
        """
        if not self.webhook_url:
            logging.warning("WEBHOOK_URL not set. Skipping sending data via webhook.")
            return None

        try:
            # Construct the message content by concatenating information from all papers.
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            message_text = f"{today} Arxiv papers summary for {workflow_name}:\n\n"
            for paper_data in data_list:
                message_text += f"Title: {paper_data['title']}\n"
                message_text += f"{paper_data['translated_title']}\n"
                message_text += f"Authors: {paper_data['authors']}\n"
                if paper_data.get("affiliations"):
                    message_text += f"Affiliations: {paper_data['affiliations']}\n"
                message_text += f"URL: {paper_data['url']}\n"
                message_text += f"Keywords: {', '.join(paper_data['keywords'])}\n"
                if paper_data.get("code"):
                    message_text += f"Code: {paper_data['code']}\n"
                if paper_data.get("demo"):
                    message_text += f"Demo: {paper_data['demo']}\n"
                if "relevance" in paper_data:  # Add relevance if it exists
                    relevance_map = {0: "Low", 1: "Medium", 2: "High"}
                    message_text += (
                        f"Relevance: {relevance_map.get(paper_data['relevance'], 'N/A')}\n"
                    )
                message_text += f"Summary: {paper_data['summary']}\n\n"
            # Remove the trailing newline characters
            message_text = message_text.rstrip("\n")

            # Construct the message payload.
            content = {"text": message_text}

            payload = {"msg_type": "text", "content": content}

            # Convert the payload to JSON.
            json_payload = json.dumps(payload)

            # Send the request to the webhook.
            headers = {"Content-Type": "application/json"}
            logging.info(
                f"Sending {len(data_list)} papers to webhook for category {workflow_name}..."
            )
            response = requests.post(self.webhook_url, data=json_payload, headers=headers)

            # Check the response status code.
            if response.status_code == 200:
                logging.info(
                    f"Successfully sent data for {len(data_list)} papers in batch '{workflow_name}'."
                )
                return response
            else:
                logging.error(
                    f"Error sending data for batch '{workflow_name}'. Status code: {response.status_code}. Response text: {response.text}"
                )
                return None

        except Exception as e:
            logging.error(
                f"An error occurred while sending webhook for batch '{workflow_name}': {e}"
            )
            return None

    def run(
        self,
        categories: str | list[str],
        max_papers_split: int = 10,
        user_interest: str | None = None,
        filter_level: str = "none",
        workflow_name: str = "",
    ):
        logging.info(
            f"Starting Arxiv summarization for categories: {categories} with workflow {workflow_name}"
        )
        papers = self.process_arxiv_url(categories, user_interest, filter_level)
        if not papers:
            logging.warning("Processing failed or no papers were found. Exiting.")
            return  # Exit gracefully if no papers or error during processing

        # Sort papers by relevance if user_interest is set
        if user_interest:
            papers.sort(key=lambda x: x.get("relevance", 0), reverse=True)
            logging.info(f"Sorted {len(papers)} papers by relevance (descending).")

        if self.webhook_url:
            num_splits = (len(papers) + max_papers_split - 1) // max_papers_split
            split_size = (len(papers) + num_splits - 1) // num_splits
            papers_split = [papers[i : i + split_size] for i in range(0, len(papers), split_size)]
            for i, papers in enumerate(papers_split):
                if len(papers_split) == 1:
                    suffix = ""
                else:
                    suffix = f" ({i+1}/{len(papers_split)})"

                self.send_arxiv_data_via_webhook(papers, workflow_name + suffix)
        else:
            logging.info("Webhook URL not configured. Papers will not be sent.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize Arxiv papers.")
    parser.add_argument(
        "category",
        nargs="+",
        default=["eess.AS"],
        help="Arxiv category or categories (default: eess.AS). Provide multiple categories separated by spaces.",
    )
    parser.add_argument(
        "--max_papers_split",
        type=int,
        default=10,
        help="Maximum number of papers to send in a single webhook request (default: 10)",
    )
    parser.add_argument(
        "--user_interest",
        type=str,
        default=None,
        help="User's area of interest for relevance evaluation (e.g., 'machine learning, NLP'). If not provided, all papers will have relevance 0.",
    )
    parser.add_argument(
        "--filter_level",
        type=str,
        default="none",
        choices=["low", "mid", "high", "none"],
        help="Filter papers based on relevance: 'low' (score >=0), 'mid' (score >=1), 'high' (score >=2), 'none' (no filtering). Default: 'none'.",
    )
    parser.add_argument(
        "--workflow_name",
        type=str,
        help="Name of the workflow for logging purposes.",
    )

    args = parser.parse_args()
    logging.info(
        f"Running Arxiv summarizer with categories: {args.category}, max_papers_split: {args.max_papers_split}, user_interest: {args.user_interest}, filter_level: {args.filter_level}, workflow_name: {args.workflow_name}"
    )

    try:
        summarizer = ArxivSummarizer()
        summarizer.run(
            args.category,
            args.max_papers_split,
            args.user_interest,
            args.filter_level,
            args.workflow_name,
        )
    except ValueError as e:
        logging.critical(f"Configuration error: {e}. Please check your .env file.")
    except Exception as e:
        logging.critical(f"An unrecoverable error occurred during execution: {e}", exc_info=True)
