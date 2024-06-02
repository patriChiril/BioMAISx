# %%
import json
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split

# fake
nlp = spacy.load("en_core_web_sm")

REPO_ROOT = Path(__file__).resolve().parent.parent
agreement_directory = REPO_ROOT / "annotation" / "review"
final_directory = REPO_ROOT / "annotation" / "best_quotes"

# %%
df = pd.read_csv(REPO_ROOT / "annotation" / "batch-10.csv", sep="\t")


# %%


def tokenize_and_convert_indices(row):
    # if row["extraction_score"] == "Perfect":
    #     row["start"] = row["start_extracted"]
    #     row["end"] = row["end_extracted"]
    doc = nlp(row["paragraph_text"])
    num_sentences = len([sent for sent in doc.sents])
    try:
        start_char = int(row["start"])
    except:
        print(row)
    if row["paragraph_text"][start_char] == " ":
        start_char += 1
    end_char = row["end"]
    start_predicted_char = row["start_extracted"]
    end_predicted_char = row["end_extracted"]

    start_token_idx = None
    end_token_idx = None
    end_predicted_token_idx = None
    start_predicted_token_idx = None

    # Iterate through tokens
    for token in doc:
        if start_char >= token.idx and start_char < token.idx + len(token):
            start_token_idx = token.i
        if end_char >= token.idx and end_char <= token.idx + len(token):
            end_token_idx = token.i
        if (
            start_predicted_char >= token.idx
            and start_predicted_char < token.idx + len(token)
        ):
            start_predicted_token_idx = token.i
        if end_predicted_char >= token.idx and end_predicted_char <= token.idx + len(
            token
        ):
            end_predicted_token_idx = token.i

    # Exclude leading and trailing punctuation tokens
    while start_token_idx is not None and doc[start_token_idx].is_punct:
        start_token_idx += 1
    while end_token_idx is not None and doc[end_token_idx].is_punct:
        end_token_idx -= 1
    while (
        start_predicted_token_idx is not None
        and doc[start_predicted_token_idx].is_punct
    ):
        start_predicted_token_idx += 1
    while end_predicted_token_idx is not None and doc[end_predicted_token_idx].is_punct:
        end_predicted_token_idx -= 1

    if end_token_idx is not None:
        ground_truth_clean_quote = doc[start_token_idx : end_token_idx + 1]
    else:
        ground_truth_clean_quote = None

    if end_predicted_token_idx is not None:
        extracted_clean_quote = doc[
            start_predicted_token_idx : end_predicted_token_idx + 1
        ]
    else:
        extracted_clean_quote = None

    return (
        start_token_idx,
        end_token_idx,
        ground_truth_clean_quote,
        start_predicted_token_idx,
        end_predicted_token_idx,
        extracted_clean_quote,
        num_sentences,
    )


# %%


def check_for_overlaps(dict_list):
    """
    Compare a list of dictionaries to find pairs where one dictionary's 'start' and 'end'
    values are contained within another dictionary's 'start' and 'end' values.

    Args:
        dict_list (list): A list of dictionaries, where each dictionary should have the
                          structure {"value": {"start": int, "end": int}}.

    Returns:
        list: A list of pairs of dictionaries that meet the containment condition.
              Each pair is represented as a tuple (dict1, dict2), where dict1's values
              are contained within dict2's values, or vice versa.
    """
    result_list = []
    for i in range(len(dict_list)):
        for j in range(i + 1, len(dict_list)):
            dict1, dict2 = dict_list[i], dict_list[j]
            start1, end1 = dict1["value"]["start"], dict1["value"]["end"]
            start2, end2 = dict2["value"]["start"], dict2["value"]["end"]

            if start1 >= start2 - 1 and end1 <= end2 + 1:
                result_list.append((dict1, dict2))
            elif start2 >= start1 - 1 and end2 <= end1 + 1:
                result_list.append((dict2, dict1))

    return result_list


def extract_entity_from_task(name_to_instances: dict):
    """Identify the entity from the task

    Motivation:
        from_name for both manual 'proposed_entity's and 'quote's is labeled
        as 'quote'. To differentiate:
            (1) If 'sentiment' is not labelled, then entity does not matter.
            (2) If there is only 1 'quote' and no 'proposed_entity', then 'quote'
                is the entity
            (3) If there are multiple 'quote's then any that are contained by another
                are the entity
            (4) If there is a 'quote' and a 'proposed_entity' and a 'sentiment' ????
    """
    if "sentiment" not in name_to_instances:
        return None, None, None
    quote_instances = len(name_to_instances.get("quote", []))
    proposed_instances = len(name_to_instances.get("proposed_entity", []))
    # substrings = check_for_overlaps(quote_instances + proposed_instances)
    if quote_instances > 1:
        sub_quotes = check_for_overlaps(name_to_instances.get("quote", []))
        if len(sub_quotes) == 1:
            substring, larger_string = sub_quotes[0]
            entity_values = substring["value"]
        elif len(sub_quotes) == 0 and proposed_instances == 1:
            entity_values = name_to_instances.get("proposed_entity")[0]["value"]
        else:
            message = "There are multiple quotes that don't contain each other and not a single proposed instance"
            raise AssertionError(message)
    elif quote_instances == 1 and proposed_instances > 0:
        # just assume the proposed instance is correct
        entity_values = name_to_instances.get("proposed_entity")[0]["value"]

        # check if each proposed is a substring?
    elif quote_instances == 1 and proposed_instances == 0:
        # quote is really a proposed entity
        entity_values = name_to_instances.get("quote")[0]["value"]
    elif quote_instances == 0 and proposed_instances == 1:
        entity_values = name_to_instances["proposed_entity"][0]["value"]
    elif "null" in name_to_instances:
        return "null", None, None
    elif proposed_instances == 2:
        # who do we pick?
        raise AssertionError("multiple proposed entities")
    else:
        raise AssertionError("unknown how to proceed")

    if "text" not in entity_values:
        raise AssertionError("'text' not in entity values")
    return entity_values["text"], entity_values["start"], entity_values["end"]


def convert_to_dictionary(annotations):
    """Convert a LS 'annotations' to dict of 'from_name' tags to list of instances"""
    from_name_to_instances = {}
    for result in annotations:
        if not from_name_to_instances.get(result["from_name"], False):
            from_name_to_instances[result["from_name"]] = [result]
        else:
            from_name_to_instances[result["from_name"]].append(result)

    sentiments = len(from_name_to_instances.get("sentiment", []))
    entity_types = len(from_name_to_instances.get("entity-type", []))
    second_sentiments = len(from_name_to_instances.get("second-sentiment", []))
    confidences = len(from_name_to_instances.get("confidence", []))
    aspects = len(from_name_to_instances.get("aspect", []))
    # there should be at most one sentiment and entity type
    assert sentiments <= 1, "too many sentiments"
    assert entity_types <= 1, "too many entity types"
    assert second_sentiments <= 1, "too many second-sentiments"
    assert confidences <= 1, "too many confidences"
    assert aspects <= 2, "too many aspects"

    # if there is a sentiment, there should be entity type
    if sentiments == 1:
        assert entity_types == 1, "no entity type"
        assert aspects >= 1, "no aspects labelled"
    if second_sentiments == 1:
        assert (
            len(from_name_to_instances["aspect"][0]["value"]["choices"]) == 2
        ), "missing second aspect with a 'second-sentiment'"

    return from_name_to_instances


def get_absa_results(task_results: list):
    annotations = task_results["annotations"][0]["result"]
    paragraph_text = task_results["data"]["paragraph_text"]
    quote_id = task_results["data"]["quote_id"]
    try:
        from_name_to_instances = convert_to_dictionary(annotations)
    except AssertionError as e:
        print(quote_id, e)
        return None
    if "sentiment" not in from_name_to_instances:
        return None
    try:
        entity, start_idx, end_idx = extract_entity_from_task(from_name_to_instances)
    except AssertionError as e:
        print(quote_id, e)
        return None
    entity_type = from_name_to_instances["entity-type"][0]["value"]["choices"][0]

    # get sentiment and aspect -- complicated due to possibiity of two
    sentiments = []
    sentiment = from_name_to_instances["sentiment"][0]
    sentiments.append(sentiment["value"]["choices"][0])
    if "second-sentiment" in from_name_to_instances:
        second_sentiment = from_name_to_instances["second-sentiment"][0]
        sentiments.append(second_sentiment["value"]["choices"][0])
    aspects = from_name_to_instances["aspect"][0]["value"]["choices"]
    # create return list of dicts
    output = []
    for sentiment, aspect in zip(sentiments, aspects):
        formatted_results = {
            "quote_id": quote_id,
            "entity": entity,
            "entity_start_idx": start_idx,
            "entity_end_idx": end_idx,
            "entity_type": entity_type,
            "aspect": aspect,
            "sentiment": sentiment,
            "paragraph_text": paragraph_text,
        }
        output.append(formatted_results)
    return output


def get_quotation_results(task_results):
    quote_id = task_results["data"]["quote_id"]
    extracted_quote = task_results["data"]["text"]
    paragraph_text = task_results["data"]["paragraph_text"]
    annotations = task_results["annotations"][0]["result"]

    (
        extraction_score,
        start,
        end,
    ) = (
        None,
        None,
        None,
    )
    quote_chunks = 0
    for result in annotations:
        if result["from_name"] == "extraction-score":
            extraction_score = result["value"]["choices"][0]
        elif result["from_name"] == "quote" and "Quote" in result["value"].get(
            "labels", []
        ):
            start = result["value"]["start"]
            end = result["value"]["end"]
            quote_chunks += 1
    if extraction_score in ["Good", "Poor", "Perfect"]:
        if start is None or end is None or start is np.nan or end is np.nan:
            raise AssertionError(
                f"{quote_id} - Extraction score present, but start or end are none"
            )
    return {
        "quote_id": quote_id,
        "extraction_score": extraction_score,
        "extracted_quote": extracted_quote,
        "paragraph_text": paragraph_text,
        "start": start,
        "end": end,
        "multiple_quote_chunks": quote_chunks > 1,
    }


def create_absa_results_df(label_studio_results: list) -> pd.DataFrame:
    results = []
    for task in label_studio_results:
        task_formatted = get_absa_results(task)
        if task_formatted:
            results.extend(get_absa_results(task))
    return pd.DataFrame(results)


def create_quotes_results_df(label_studio_results: list) -> pd.DataFrame:
    results = []
    for task in label_studio_results:
        try:
            results.append(get_quotation_results(task))
        except AssertionError as e:
            print(e)
    quote_results = pd.DataFrame(results)

    quote_results = quote_results[
        quote_results["extraction_score"].isin(["Good", "Poor", "Perfect"])
        & ~quote_results["multiple_quote_chunks"]
    ]
    quote_results["start_extracted"] = quote_results.apply(
        lambda row: row["paragraph_text"].find(row["extracted_quote"]), axis=1
    )
    quote_results["end_extracted"] = quote_results.apply(
        lambda row: len(row["extracted_quote"]) + row["start_extracted"], axis=1
    )
    assert quote_results["start_extracted"].isna().sum() == 0
    quote_results[
        [
            "start_token_ground_truth",
            "end_token_ground_truth",
            "cleaned_ground_truth_quote",
            "start_token_predicted",
            "end_token_predicted",
            "cleaned_predicted_quote",
            "num_sentences",
        ]
    ] = quote_results.apply(tokenize_and_convert_indices, axis=1, result_type="expand")
    return quote_results


def format_dfs(df: pd.DataFrame, max_quote_id=None):
    df = df.set_index(["quote_id", "entity", "aspect_number"])
    df = df[df.index.get_level_values("quote_id") <= max_quote_id]
    df = df.fillna("NA")
    return df


def create_df_from_directory(directory: Path, func) -> pd.DataFrame:
    """Concatenate dfs resulting from aplying func to all files in directory"""
    all_results = []
    for output_json in directory.iterdir():
        # load in label studio jsons, format as dfs and merge
        with open(directory / output_json, "r") as f:
            results = json.load(f)
        annotation_results = func(results)
        all_results.append(annotation_results)

    complete_results = pd.concat(all_results)
    return complete_results


def eliminate_duplicate_quotes(df):
    """Filter all duplicated quotes from dataset"""
    unique_quote_ids = df.drop_duplicates(subset="quote_id")
    duplicates = unique_quote_ids[unique_quote_ids.duplicated(subset="quote_string")]
    print(f"{len(duplicates)} duplicates found")
    df = df[~df["quote_id"].isin(duplicates["quote_id"].unique())]
    return df


def find_values_above_threshold(df: pd.DataFrame, n: int) -> list[(str, str)]:
    """Given a df, return all row, column pairs by name with value > n"""
    name_pairs = []
    for column in df.columns:
        for index, value in df[df[column] > n].iterrows():
            name_pairs.append((index, column))
    return name_pairs


def join_and_clean_dataset(
    absa_results: pd.DataFrame, quote_results: pd.DataFrame
) -> pd.DataFrame:
    """Join labeled quotes with the absa results dataframe"""
    quote_results = quote_results.drop_duplicates(subset=["quote_id"])
    quote_results = quote_results.set_index("quote_id")
    cleaned_dataset = absa_results.merge(quote_results, on="quote_id", how="left")
    cleaned_dataset = cleaned_dataset[
        ~cleaned_dataset["cleaned_ground_truth_quote"].isna()
    ]
    cleaned_dataset["quote_string"] = cleaned_dataset[
        "cleaned_ground_truth_quote"
    ].apply(lambda quote: quote.text)

    # %%
    aspects_to_remove = [
        "Ease of Use",
        "Economincs",
        "General",
        "Quality",
        "Responsiveness",
    ]
    # entity_types_to_remove = ["Crop", "Organization"]

    # cleaned_dataset = cleaned_dataset[
    #     ~cleaned_dataset["entity_type"].isin(entity_types_to_remove)
    # ]
    # cleaned_dataset = cleaned_dataset[
    #     ~cleaned_dataset["aspect"].isin(aspects_to_remove)
    # ]
    cleaned_dataset = eliminate_duplicate_quotes(cleaned_dataset)
    return cleaned_dataset


def filter_dataset(df: pd.DataFrame, min_instances=50) -> pd.DataFrame:
    """Eliminate samples with under min_instances"""
    pivot_table = df.pivot_table(
        index="aspect", columns="entity_type", aggfunc="size", fill_value=0
    )
    entity_type_aspect_pairs = find_values_above_threshold(pivot_table, min_instances)
    print(entity_type_aspect_pairs)
    filtered_dataset = df[
        df.apply(
            lambda row: (row["aspect"], row["entity_type"]) in entity_type_aspect_pairs,
            axis=1,
        )
    ]
    return filtered_dataset


def assign_sets(df, train, test, validation):
    def assign_by_row(row):
        if row["quote_id"] in train:
            return "train"
        elif row["quote_id"] in test:
            return "test"
        elif row["quote_id"] in validation:
            return "validation"
        else:
            print(row)
            return "unknown"

    df["set"] = df.apply(assign_by_row, axis=1)
    return df


def format_data_for_mvp(df, instance_samples=None, quad=False):
    """df should have 'entity_type',... 'set'"""

    def format_quoutes_mvp(group):
        quote = " ".join(
            [token.text for token in group["cleaned_ground_truth_quote"].iloc[0]]
        )
        quote = quote.replace("\n", "")
        entities = group.apply(
            lambda row: (
                row["entity"],
                f'{row["entity_type"]} {row["aspect"]}',
                row["sentiment"],
                "NULL",  # if quad!
            ),
            axis=1,
        ).tolist()
        return f"{quote}####{entities}"

    if quad:
        output_directory = REPO_ROOT / "annotation" / "datasets" / "quad"
    else:
        output_directory = REPO_ROOT / "annotation" / "datasets" / "mvp"
    output_directory.mkdir(parents=True, exist_ok=True)
    paper_names = {
        "train": "train",
        "test": "test",
        "validation": "dev",
    }
    for set in ["train", "test", "validation"]:
        temp_df = df[df["set"] == set]
        temp_df = temp_df[temp_df["sentiment"] != "Conflict"]
        if quad:
            temp_df["sentiment"] = temp_df["sentiment"].apply(lambda s: s.lower())
        if instance_samples is not None and set == "train":
            allowed_quote_ids = temp_df.groupby(["entity_type", "aspect"]).head(
                instance_samples
            )["quote_id"]
        else:
            allowed_quote_ids = temp_df["quote_id"]
        result = (
            temp_df.groupby("quote_id")
            .apply(format_quoutes_mvp)
            .reset_index(name="formatted_quotes")
        )
        result = result[result["quote_id"].isin(allowed_quote_ids)]
        with open(output_directory / f"{paper_names[set]}.txt", "w+") as f:
            f.write("\n".join(result["formatted_quotes"]))


def format_data_for_InstructABSA(df, instances=None):
    """df should have 'entity_type',... 'set'"""

    def format_dataframe_InstructABSA(df):
        # Create a DataFrame to hold the transformed data
        transformed_data = []

        for _, group in df.groupby("quote_id"):
            sentence_id = group["quote_id"].iloc[0]
            raw_text = group["quote_string"].iloc[0]
            aspect_terms = []

            for index, row in group.iterrows():
                aspect_term = {"term": row["entity"], "polarity": row["sentiment"]}
                aspect_terms.append(aspect_term)

            transformed_data.append([sentence_id, raw_text, aspect_terms])

        # Create the DataFrame with the desired format
        transformed_df = pd.DataFrame(
            transformed_data, columns=["sentenceId", "raw_text", "aspectTerms"]
        )

        # Add the aspectCategories column
        transformed_df["aspectCategories"] = (
            "[{'category': 'noaspectcategory', 'polarity': 'none'}]"
        )

        return transformed_df

    output_directory = REPO_ROOT / "annotation" / "datasets" / "InstructABSA"
    output_directory.mkdir(parents=True, exist_ok=True)
    paper_names = {
        "train": "mbio_Train",
        "test": "mbio_Test",
        "validation": "mbio_Validation",
    }
    for set in ["train", "test", "validation"]:
        temp_df = df[df["set"] == set]
        result = format_dataframe_InstructABSA(temp_df)
        result = result.set_index("sentenceId")
        if instances is not None and set == "train":
            # Shuffle the DataFrame
            shuffled_df = result.sample(
                frac=1, random_state=42
            )  # Use a random_state for reproducibility

            # Create N disjoint samples of K rows
            samples = []
            for i in range(10):
                sample = shuffled_df.iloc[i * instances : (i + 1) * instances]
                samples.append(sample)
            for i, sample in enumerate(samples):
                sample.to_csv(output_directory / f"{paper_names[set]}-{i}.csv")
        else:
            result.to_csv(output_directory / f"{paper_names[set]}.csv")


def find_token_index_by_char_index(doc, char_index, start_of_span=False):
    for i, token in enumerate(doc):
        # if labeler selected preceeding whitespace, include it
        if start_of_span and i > 0:
            start = token.idx - len(doc[i - 1].whitespace_)
        else:
            start = token.idx
        if not start_of_span:
            end = token.idx + len(token.text) + len(token.whitespace_)
        else:
            end = token.idx + len(token.text)
        if char_index >= start and char_index < end:
            return token.i  # token.i is from start of paragraph
    return token.i  # token.i is from start of paragraph


def format_data_for_BARTABSA_set(df):
    transformed_data = []

    polartiy_names = {
        "Positive": "POS",
        "Negative": "NEG",
        "Neutral": "NEU",
        "Conflict": "CON",
    }

    for _, group in df.groupby("quote_id"):
        quote_id = int(group["quote_id"].iloc[0])
        if quote_id == 21011:
            print("we got em")
        raw_words = " ".join(
            [tok.text for tok in group["cleaned_ground_truth_quote"].iloc[0]]
        )
        cleaned_ground_truth_quote = group["cleaned_ground_truth_quote"].iloc[0]

        words = [token.text for token in cleaned_ground_truth_quote]
        aspect_dicts = []
        opinions = [{"term": []}]

        for _, row in group.iterrows():
            start_idx = row["entity_start_idx"]
            end_idx = row["entity_end_idx"]
            entity_start_token_idx = find_token_index_by_char_index(
                cleaned_ground_truth_quote, start_idx, start_of_span=True
            )
            entity_end_token_idx = (
                find_token_index_by_char_index(cleaned_ground_truth_quote, end_idx) + 1
            )
            polarity = polartiy_names[row["sentiment"]]
            entity_start_idx_relative_to_quote = (
                entity_start_token_idx - row["start_token_ground_truth"]
            )
            entity_end_idx_relative_to_quote = (
                entity_end_token_idx - row["start_token_ground_truth"]
            )
            aspect_dict = {
                "from": int(
                    entity_start_idx_relative_to_quote
                ),  # entity_start_token is from start of whole paragraph
                "to": int(entity_end_idx_relative_to_quote),
                "polarity": polarity,
                "term": [
                    token.text
                    for token in cleaned_ground_truth_quote[
                        entity_start_idx_relative_to_quote:entity_end_idx_relative_to_quote
                    ]
                ],
            }

            aspect_dicts.append(aspect_dict)

        transformed_dict = {
            "raw_words": raw_words,
            "words": words,
            "task": "AE-OE",
            "aspects": aspect_dicts,
            "opinions": opinions,
            "quote_id": quote_id,
        }

        transformed_data.append(transformed_dict)

    return transformed_data


def format_data_for_BARTABSA(df):
    output_directory = REPO_ROOT / "annotation" / "datasets" / "BARTABSA"
    output_directory.mkdir(parents=True, exist_ok=True)
    paper_names = {
        "train": "train_convert",
        "test": "test_convert",
        "validation": "dev_convert",
    }
    for set in ["train", "test", "validation"]:
        temp_df = df[df["set"] == set]
        result = format_data_for_BARTABSA_set(temp_df)
        with open(output_directory / f"{paper_names[set]}.json", "w+") as f:
            json.dump(result, f)


# %%
# deprecated
def prepare_quote_data(quote_results: pd.DataFrame):
    """Prepare quote_results for performance computations"""
    comparable_results = quote_results[
        quote_results["extraction_score"].isin(["Good", "Poor"])
        & ~quote_results["multiple_quote_chunks"]
    ]
    print(len(comparable_results))
    comparable_results = comparable_results.dropna(subset=["start", "end"])
    print(len(comparable_results))
    comparable_results["start_extracted"] = comparable_results.apply(
        lambda row: row["paragraph_text"].find(row["extracted_quote"]), axis=1
    )
    comparable_results["end_extracted"] = comparable_results.apply(
        lambda row: len(row["extracted_quote"]) + row["start_extracted"], axis=1
    )
    comparable_results[
        [
            "start_token_ground_truth",
            "end_token_idx_ground_truth",
            "cleaned_ground_truth_quote",
            "start_token_predicted",
            "end_token_predicted",
            "cleaned_predicted_quote",
        ]
    ] = comparable_results.apply(
        tokenize_and_convert_indices, axis=1, result_type="expand"
    )
    return comparable_results


# %%
def precision(gold: list[tuple], prediction: list[tuple]) -> float:
    """caclulate P as defined in https://aclanthology.org/D13-1101.pdf"""
    total = 0
    for g, p in zip(gold, prediction):
        total += overlap(g, p)
    return total / len(prediction)


def recall(gold: list[tuple], prediction: list[tuple]) -> float:
    """caclulate R as defined in https://aclanthology.org/D13-1101.pdf"""
    total = 0
    for g, p in zip(gold, prediction):
        total += overlap(p, g)
    return total / len(gold)


def f_score(gold: list[tuple], prediction: list[tuple]) -> float:
    R = recall(gold, prediction)
    P = precision(gold, prediction)
    return 2 * P * R / (P + R)


def overlap(x, y):
    # Calculate the overlap length
    overlap_length = min(x[1], y[1]) - max(x[0], y[0]) + 1

    if overlap_length <= 0:
        return 0.0

    # Calculate the length of y
    y_length = y[1] - y[0] + 1

    # Calculate the proportion of tokens in y that are overlapped by x
    proportion_overlap = overlap_length / y_length

    return proportion_overlap


def missing_first_token(gold, predictions):
    missing_first = 0
    for g, p in zip(gold, predictions):
        if (g[1] == p[1]) and (g[0] == (p[0] - 1)):
            missing_first += 1
    return missing_first


def calculate_metrics(df: pd.DataFrame):
    """Calculate metrics defined in https://aclanthology.org/D13-1101.pdf

    Args:
        df: should have 'start_token_ground_truth', 'end_token_idx_ground_truth'
            'start_token_predicted', 'end_token_predicted'
    """
    # create tuples
    gold = [
        (row["start_token_ground_truth"], row["end_token_idx_ground_truth"])
        for _, row in df.iterrows()
    ]
    predictions = [
        (row["start_token_predicted"], row["end_token_predicted"])
        for _, row in df.iterrows()
    ]
    print(f_score(gold, predictions))
    print(precision(gold, predictions))
    print(recall(gold, predictions))
    print(missing_first_token(gold, predictions))
    print(len(gold))


if __name__ == "__main__":
    absa_results = create_df_from_directory(final_directory, create_absa_results_df)

    quote_results = create_df_from_directory(final_directory, create_quotes_results_df)

    cleaned_dataset = join_and_clean_dataset(absa_results, quote_results)
    cleaned_dataset.to_csv("biomaisx.csv")

    filtered_dataset = filter_dataset(cleaned_dataset, min_instances=50)
    temp_df = filtered_dataset.groupby("quote_id")[["entity_type", "aspect"]].first()

    X = temp_df.index
    y = temp_df[["entity_type", "aspect"]]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_test, X_validation, y_test, y_validation = train_test_split(
        X_temp, y_temp, test_size=0.33, stratify=y_temp, random_state=42
    )

    formatted_data = assign_sets(filtered_dataset, X_train, X_test, X_validation)

    filtered_dataset.groupby(["entity_type", "aspect", "set"]).count()

    print(
        f"Filtered dataset has: {filtered_dataset.groupby('quote_id')['num_sentences'].last().sum()} sentences"
    )

    print("formatting data")
    format_data_for_BARTABSA(formatted_data)
    print("formatted data for bartabsa")
    format_data_for_InstructABSA(formatted_data, instances=20)
    print("formatted data for instructabsa")
    format_data_for_mvp(formatted_data, instance_samples=None, quad=True)
