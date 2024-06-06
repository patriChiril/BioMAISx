import pandas as pd
from pathlib import Path
import logging
from textacy import preprocessing as preproc
import fastavro
import re
from transformers import pipeline
from torch.cuda import is_available
import json
import numpy as np

repo_root = Path(__file__).resolve().parent


def dataframe_from_avro(file_path: str) -> pd.DataFrame:
    """create dataframe from avro file

    source: https://towardsdatascience.com/csv-files-for-storage-absolutely
        -not-use-apache-avro-instead-7b7296149326
    """
    # 1. List to store the records
    avro_records = []
    # 2. Read the Avro file
    with open(file_path, "rb") as fo:
        avro_reader = fastavro.reader(fo)
        for record in avro_reader:
            avro_records.append(record)
    # 3. Convert to pd.DataFrame
    df_avro = pd.DataFrame(avro_records)
    return df_avro


def standarize_articles_dataframe(articles_df: pd.DataFrame) -> pd.DataFrame:
    """Perform basic standardization of text and column names"""
    publishers = pd.read_csv(repo_root / "publishers.csv", index_col=0)
    articles_df = articles_df.loc[
        articles_df["source_code"].isin(publishers.index)
    ].copy()
    logging.info(f"Length after articles filtered by publisher: {len(articles_df)}")
    # clean 'text' column which is concatentation of 'snippet' and 'body'
    articles_df["text_raw"] = articles_df["snippet"] + "\n" + articles_df["body"]
    # html tags
    articles_df["text"] = articles_df["text_raw"].str.replace(r"<.*>", "", regex=True)
    # check word count manually, remove articles with fewer than 50 words
    MIN_WORDS = 50
    articles_df["manual_word_count"] = articles_df["text"].str.split().str.len()
    articles_df = articles_df[articles_df["manual_word_count"] >= MIN_WORDS]
    logging.info(f"Length after <{MIN_WORDS} word articles removed: {len(articles_df)}")

    articles_df = articles_df.rename(columns={"an": "id", "language_code": "language"})
    articles_df = articles_df[
        [
            "id",
            "text_raw",
            "text",
            "title",
            "publisher_name",
            "publication_datetime",
            "language",
            "byline",
        ]
    ]
    # normalize whitespaces
    articles_df["text"] = articles_df["text"].apply(
        lambda x: preproc.normalize.whitespace(str(x))
    )
    # normalize quotes
    articles_df["text"] = articles_df["text"].apply(
        lambda x: preproc.normalize.quotation_marks(str(x))
    )
    return articles_df


def open_dataset(dataset_path: Path) -> pd.DataFrame:
    """Read dataset into dataframe
    
    Args:
        dataset_path: path to dataset. Must be avro or csv
    """
    logging.info(f"Starting pipeline on {dataset_path.stem}")
    if dataset_path.suffix == ".avro":
        raw_articles_df = dataframe_from_avro(dataset_path)
    elif dataset_path.suffix == ".csv":
        raw_articles_df = pd.read_csv(dataset_path)
    else:
        raise ValueError("Articles must be .avro or .csv")
    return raw_articles_df
    

def filter_dataset(articles_df: pd.DataFrame, keyword_filters: list[dict]) -> pd.DataFrame:
    """Filters articles_df to only rows containing certain strings
    
    Args:
        articles_df: dataframe representing articles
        keyword_filters: list of dictionaries with 'column_name' and 'keywords' keys.
            The value of 'column_name' must be a string the corresponds to a column
            in articles_df that has string values. keywords should be a list of words.
            Only those rows that contain one of these words in the specified column name
            will be kept
    """
    for keywords_filter in keyword_filters:
        articles_df = articles_df[
            articles_df[keywords_filter["column_name"]].str.contains(
                "|".join(keywords_filter["keywords"]), regex=True
            )
        ]
    logging.info(f"Dataset contains {len(articles_df)} articles after filtering.")
    return articles_df


def get_relevant_articles_from_directory(directory_of_datasets: Path, keyword_filters: list[dict]) -> pd.DataFrame:
    """Read all files in directory as tables, saving relevant relevant rows as DataFrame
    
    Args:
        directory_of_datasets: path to directory containing .avro and .csvs
        keyword_filters: list of dictionaries with 'column_name' and 'keywords' keys.
            The value of 'column_name' must be a string the corresponds to a column
            in articles_df that has string values. keywords should be a list of words.
            Only those rows that contain one of these words in the specified column name
            will be kept
    """
    all_filtered_articles = []
    for dataset_path in Path(directory_of_datasets).resolve().iterdir():
        raw_df = open_dataset(dataset_path)
        articles_df = standarize_articles_dataframe(raw_df)
        filtered_articles = filter_dataset(articles_df, keyword_filters)
        all_filtered_articles.append(filtered_articles)
    return pd.concat(all_filtered_articles)


def extract_quotations_from_paragraph(text: str, classifier) -> list:
    """Extract a list of quoatations from the selected text"""
    predictions = classifier(text)
    quotations = []  # [(start, stop)]
    speakers = []  # [(start, stop)]
    for prediction in predictions:
        entity = prediction["entity"]
        if entity == "Out":
            continue
        if entity == "B-Speaker" or speakers == []:
            speakers.append([prediction["start"], prediction["end"]])
        elif entity == "I-Speaker":
            speakers[-1][-1] = prediction["end"]
        elif entity.startswith("B-") or quotations == []:
            quotations.append([prediction["start"], prediction["end"]])
        elif entity.startswith("I-"):
            quotations[-1][-1] = prediction["end"]

    final_quotations = [text[quote[0]: quote[1]] for quote in quotations]
    return final_quotations


def split_text_into_paragraphs(text: str) -> list[str]:
    """Split a text into logical subsections for model processing"""
    paragraphs = re.split(r"[.?!'\"][\n\r]", text)
    paragraphs = list(filter(lambda p: p.strip() != "", paragraphs))
    return paragraphs


def extract_quotations_from_text(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract quotations from the 'paragraph_text' column of each row in df"""
    # first create df with all paragraphs
    full_dataset = pd.DataFrame()
    for _, row in df.iterrows():
        paragraphs_list = split_text_into_paragraphs(row["paragraph_text"])
        quotes_df = pd.DataFrame(
            data={"paragraph_text": paragraphs_list}
        )
        quotes_df["full_text"] = row["paragraph_text"]
        full_dataset = pd.concat([full_dataset, quotes_df])   
    device = 0 if is_available() else -1
    classifier = pipeline("ner", model="Iceland/quote-model-BERTm-v1", device=device)
    full_dataset["quote_text"] = full_dataset["paragraph_text"].progress_apply(extract_quotations_from_paragraph, args=(classifier,))
    # one row for each quote
    full_dataset = full_dataset.explode('quote_text', ignore_index=True).dropna(subset="quote_text")

    return full_dataset



# filter quotes
def read_files_to_list(directory_path: Path):
    all_strings = []  # Initialize an empty list to hold all strings from all files

    # Loop through each file in the specified directory
    for filename in directory_path.iterdir():
        file_path = directory_path / filename
        
        # Check if the current item is a file
        if file_path.is_file():
            with open(file_path, encoding="utf-8") as file:
                # Read each line from the file, strip newline characters, and add to list
                for line in file:
                    clean_line = line.strip()  # Remove leading/trailing whitespace
                    if clean_line:  # Add non-empty strings to the list
                        all_strings.append(clean_line)
    
    return all_strings


def filter_quotes_to_relevant(quotes_dataframe, interesting_terms):
    interesting_terms = [r"\b" + term.lower() + r"\b" for term in interesting_terms]
    quotes_dataframe = quotes_dataframe[~quotes_dataframe["quote_text"].isna()]
    quotes_dataframe = quotes_dataframe[quotes_dataframe["quote_text"].str.contains("|".join(interesting_terms), regex=True)]
    quotes_dataframe.reset_index(inplace=True)
    return quotes_dataframe


def find_noun_spans(quotes_dataframe, interesting_terms):
    # detect quotes most likely to be relevant

    spacy.prefer_gpu()

    nlp = spacy.load("en_core_web_sm")
    regexp = re.compile("|".join(interesting_terms))
    texts = quotes_dataframe.loc[:, "paragraph_text"]
    good_leads = []
    for idx, doc in enumerate(nlp.pipe(texts, disable=["ner"])):
        assert quotes_dataframe.loc[idx, "paragraph_text"] == doc.text
        good_lead = False
        chunks = []
        for ent in doc.noun_chunks:
            chunks.append((ent.start_char, ent.end_char))
            if (
                "subj" in ent.root.dep_ and 
                regexp.search(ent.text) and
                (doc.text[ent.start_char:ent.end_char+10] in quotes_dataframe.loc[idx, "quote_text"])
            ):
                good_lead = True

        quotes_dataframe.loc[idx, "noun_spans"] = str(chunks)
        quotes_dataframe.loc[idx, "relevant_subject"] = good_lead

    quotes_with_relevant_subjects = quotes_dataframe[quotes_dataframe["relevant_subject"]]
    quotes_with_relevant_subjects["noun_spans"] = quotes_with_relevant_subjects["noun_spans"].apply(literal_eval)
    return quotes_with_relevant_subjects


# break into one row for each noun span
def create_row_for_each_noun_span(quotes_with_relevant_subjects):
    """Expand a dataframe of quotes in a df of quote-noun span pairs

    """
    relevant_spans_rows = []
    for idx, row in quotes_with_relevant_subjects.iterrows():
        for noun_span in row["noun_spans"]:
            relevant_spans_rows.append(
                [idx, row["quote_text"], row["paragraph_text"], row["full_text"],
                row["article_id"], noun_span[0], noun_span[1]]
            )

    data_for_annotation = pd.DataFrame(
        data=relevant_spans_rows,
        columns=[
            "quote_id", "quote_text", "paragraph_text", "full_text", "article_id",
            "span_start", "span_end"
        ]
    )
    data_for_annotation = data_for_annotation.reset_index()
    return data_for_annotation

def add_entity_details(df):
    """Add entity text to df containing 'paragraph_text', 'span_start', and 'span_end' cols
    
    This was used to transition from 'paragraphs_of_interest' to 'mbio-annotations-v2-all' on July
    20, 2023. 
    """
    df["entity"] = df.apply(lambda row: row["paragraph_text"][row["span_start"]:row["span_end"]], axis=1)
    return df





def split_dataframe_by_unique_ids(df, x):
    """Split a DataFrame into groups with x unique quote_ids.
    
    Parameters:
    - df: The input DataFrame.
    - x: The desired number of unique quote_ids per group.
    
    Returns:
    - A list of DataFrames, each containing up to x unique quote_ids.
    """
    # Group by quote_id and store each group as a separate DataFrame
    grouped = [group for _, group in df.groupby("quote_id")]
    
    batches = []  # List to store the result batches
    current_batch = []  # Temporary storage for the current batch
    unique_ids = set()  # Track unique quote_ids in the current batch

    for group in grouped:
        quote_id = group["quote_id"].iloc[0]  # Assuming quote_id is the same for the whole group
        
        # Check if adding this group would exceed the unique quote_id limit
        if len(unique_ids) < x or quote_id in unique_ids:
            # Add the group to the current batch
            current_batch.append(group)
            unique_ids.add(quote_id)
        else:
            # Once the batch reaches x unique quote_ids, save it and start a new batch
            batches.append(pd.concat(current_batch))
            current_batch = [group]
            unique_ids = {quote_id}

    # Don't forget to add the last batch if it's not empty
    if current_batch:
        batches.append(pd.concat(current_batch))
    
    return batches


def json_format(idx, result_json, data_dict):
    return {
        "id": 1,
        "annotations": [
            {
                "id": idx,
                "result": [
                    result_json
                ],
            }
        ],
        "data": data_dict,
        "meta": {},
        "inner_id": 1,
        "total_annotations": 1,
        "cancelled_annotations": 0,
        "total_predictions": 0,
        "comment_count": 0,
        "unresolved_comment_count": 0,
        "last_comment_updated_at": None,
        "updated_by": None,
        "comment_authors": []
    }

def convert_to_json_with_entity_highlight(batch_table: pd.DataFrame) -> dict:
    jsons = []
    for idx, row in batch_table.iterrows():
        data_dict = {
            "text": row["text"],
            "paragraph_text": row["paragraph_text"],
            "full_text": row["full_text"],
            "article_id": row["article_id"],
            "quote_id": row["quote_id"],
            "entity": row["entity"],
            "span_start": row["span_start"],
            "spand_end": row["span_end"],
        }
        result_dict = {
            "value": {
                "start": row["span_start"],
                "end": row["span_end"],
                "text": row["entity"],
                "labels": [
                    "Proposed entity"
                ]
            },
            "type": "labels",
            "from_name": "proposed_entity",
            "to_name": "paragraph_text",
            "readonly": False,
            "hidden": False
        }
        jsons.append(json_format(idx, result_dict, data_dict))
    return jsons
        

def convert_raw_factiva_data_to_batches_for_annotation(directory_of_datasets: Path, lexicon_dir: Path, name_of_dataset: str, max_quotes_per_batch: int):
    all_relevant_articles = get_relevant_articles_from_directory(directory_of_datasets)
    quotes_df = extract_quotations_from_text(all_relevant_articles)
    
    all_interesting_terms = read_files_to_list(lexicon_dir)

    all_quotes_for_annotation = []
    for quotes_chunk in np.array_split(quotes_df, 10):
        quotes_data = filter_quotes_to_relevant(quotes_chunk, all_interesting_terms)
        quotes_with_noun_chunks = find_noun_spans(quotes_data, all_interesting_terms)
        quote_noun_pairs = create_row_for_each_noun_span(quotes_with_noun_chunks)
        quote_noun_pairs_entity_details = add_entity_details(quote_noun_pairs)
        all_quotes_for_annotation.append(quote_noun_pairs_entity_details)

    total_quote_set = pd.concat(all_quotes_for_annotation)
    unique_article_quote_pairs = total_quote_set.groupby(["article_id", "quote_id"])["index"].first()
    unique_article_quote_pairs = unique_article_quote_pairs.reset_index()
    unique_article_quote_pairs["unique_quote_id"] = unique_article_quote_pairs.index
    unique_article_quote_pairs = unique_article_quote_pairs.set_index(["quote_id", "article_id"])
    total_quote_set_unique = total_quote_set.merge(unique_article_quote_pairs, how="left", left_on=["quote_id", "article_id"], right_index=True)
    total_quote_set_unique = total_quote_set_unique[["unique_quote_id", "article_id", "quote_text", "paragraph_text", "full_text", "span_start", "span_end", "entity"]]
    total_quote_set_clean = total_quote_set_unique.rename(columns={"unique_quote_id": "quote_id", "quote_text": "text"})
    filtered_quote_set = total_quote_set_clean.groupby("quote_id").filter(lambda x: len(x) < 50)


    batches = split_dataframe_by_unique_ids(filtered_quote_set, max_quotes_per_batch)

    for i, batch in enumerate(batches, start=1):
        batch_json = convert_to_json_with_entity_highlight(batch)
        with Path(f"{name_of_dataset}-{i}.json").open("w") as f:
            json.dump(batch_json, f)