import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             precision_score, recall_score)

here = Path(__file__).resolve().parent


def get_annotation(task_results) -> list[dict]:
    """Transform labelstudio task result in to 
    [quote_id, entity, type, aspect, sentiment, confidence]"""
    output = []
    quote_id = task_results["data"]["quote_id"]
    entity = task_results["data"]["entity"]
    paragraph_text = task_results["data"]["paragraph_text"]
    proposed_entity = None
    entity_start = task_results["data"].get("span_start", None)
    entity_end = task_results["data"].get("span_end", None)
    entity_type, aspects, confidence = None, [None], None


    annotations = task_results["annotations"][0]["result"]
    sentiments = []
    for result in annotations:
        if result["from_name"] == "entity-type":
            entity_type = result["value"]["choices"][0]
        # second case for older annotation interface where proposed entity in labelled
        # tasks erroneously had 'from_name' as quote, but lacked the quote
        # label to are fortunately distinguishable
        if (result["from_name"] == "proposed_entity") or ((result["from_name"] == "quote") and ("labels" not in result["value"])):
            entity_start = result["value"]["start"]
            entity_end = result["value"]["end"]
            # in at least one case the text is missing for a mysterious reason
            if "text" not in result["value"]:
                proposed_entity = paragraph_text[entity_start:entity_end]
            else:
                proposed_entity = result["value"]["text"]
        elif result["from_name"] == "aspect":
            aspects = result["value"]["choices"]
        elif result["from_name"] == "sentiment":
            sentiments.insert(0, result["value"]["choices"][0])
        elif result["from_name"] == "second-sentiment":
            sentiments.append(result["value"]["choices"][0])
        elif result["from_name"] == "confidence":
            confidence = result["value"]["choices"][0]
    if sentiments == []:
        sentiments = [None]
    if len(aspects) != len(sentiments):
        # print(quote_id, entity)
        return output
    if entity_type is not None and aspects == [None]:
        pass #print("NO ASPECT!", quote_id, entity_type)
    for i, aspect in enumerate(sorted(aspects)):
        formatted_results = {
            "quote_id": quote_id,
            "entity": entity,
            "paragraph_text": paragraph_text,
            "proposed_entity": proposed_entity,
            "entity_start": entity_start,
            "entity_end": entity_end,
            "entity_type": entity_type,
            "aspect": aspect,
            "aspect_number": i,
            "sentiment": sentiments[i], 
            "confidence": confidence,
        }
        output.append(formatted_results)
    return output


def create_results_df(label_studio_results: list) -> pd.DataFrame:
    results = []
    for task in label_studio_results:
        results.extend(get_annotation(task))
    
    results_df = pd.DataFrame(results)
    results_df = results_df.dropna(subset=["entity_type", "aspect", "sentiment"])
    return results_df


def format_dfs(df: pd.DataFrame, max_quote_id=None, min_quote_id=None):
    df = df.set_index(["quote_id", "entity", "aspect_number"])
    if max_quote_id is not None:
        df = df[df.index.get_level_values("quote_id") <= max_quote_id]
    if min_quote_id is not None:
        df = df[df.index.get_level_values("quote_id") >= min_quote_id]

    df = df.fillna("NA")
    return df

def get_entity_from_span(row: pd.Series) -> str:
    if pd.isna(row["paragraph_text"]):
        print(f"No paragraph text for {row.name[0]}")
        return None
    try:
        start = int(row.name[2])
        stop = int(row.name[3])
    except ValueError:
        print(f"Start or stop index can't be cast to int for {row.name[0]}")
        return None
    return row["paragraph_text"][start:stop]

    
def create_comparison_table(df_a, df_b):
    """Given two labelstudio result dataframes, merge on matching rows for comparison
    
    Merges together two annotations if the spans of the selected entity text are from the
    same quote and overlap at all. 

    Args:
        df_a: should have quote_id, entity_start, entity_end, aspect, entity_type,
            sentiment, confidence, paragraph_text columns
        df_b: same as a
    """
    results = []
    # Combine and iterate over unique quote_id and aspect_number pairs from both DataFrames
    all_keys = pd.concat([df_a, df_b]).index.droplevel(1).drop_duplicates()
    
    for (quote_id, aspect_number) in all_keys:
        try:
            rows_a = df_a.xs((quote_id, aspect_number), level=('quote_id', 'aspect_number'))
        except KeyError:
            rows_a = pd.DataFrame()
        
        try:
            rows_b = df_b.xs((quote_id, aspect_number), level=('quote_id', 'aspect_number'))
        except KeyError:
            rows_b = pd.DataFrame()

        matched = set()
        for index_a, row_a in rows_a.iterrows():
            a_matched = False
            if (row_a["entity_start"] == "NA" or row_a["entity_end"] == "NA"):
                print(quote_id)
                continue
            for index_b, row_b in rows_b.iterrows():
                if (row_b["entity_start"] != "NA" and row_b["entity_end"] != "NA") and not (row_a['entity_end'] < row_b['entity_start'] or row_a['entity_start'] > row_b['entity_end']):
                    # Overlapping condition met
                    min_start = min(row_a['entity_start'], row_b['entity_start'])
                    max_end = max(row_a['entity_end'], row_b['entity_end'])
                    result_index = (quote_id, aspect_number, min_start, max_end)
                    result_row = {**{'entity_start_a': row_a['entity_start'], 'entity_end_a': row_a['entity_end'], 'entity_type_a': row_a['entity_type'], 'aspect_a': row_a['aspect'], 'sentiment_a': row_a['sentiment'], 'confidence_a': row_a['confidence']},
                                  **{'entity_start_b': row_b['entity_start'], 'entity_end_b': row_b['entity_end'], 'entity_type_b': row_b['entity_type'], 'aspect_b': row_b['aspect'], 'sentiment_b': row_b['sentiment'], 'confidence_b': row_b['confidence']}}
                    results.append((result_index, result_row))
                    matched.add(index_b)
                    a_matched = True
            if not a_matched:
                # Handle rows in A with no matches in B
                result_index = (quote_id, aspect_number, row_a['entity_start'], row_a['entity_end'])
                result_row = {**{'entity_start_a': row_a['entity_start'], 'entity_end_a': row_a['entity_end'], 'entity_type_a': row_a['entity_type'], 'aspect_a': row_a['aspect'], 'sentiment_a': row_a['sentiment'], 'confidence_a': row_a['confidence']},
                              **{'entity_start_b': None, 'entity_end_b': None, 'entity_type_b': None, 'aspect_b': None, 'sentiment_b': None, 'confidence_b': None}}
                results.append((result_index, result_row))
        for index_b, row_b in rows_b.iterrows():
            if index_b not in matched:
                # Handle rows in B with no matches in A
                result_index = (quote_id, aspect_number, row_b['entity_start'], row_b['entity_end'])
                result_row = {**{'entity_start_a': None, 'entity_end_a': None, 'entity_type_a': None, 'aspect_a': None, 'sentiment_a': None, 'confidence_a': None},
                              **{'entity_start_b': row_b['entity_start'], 'entity_end_b': row_b['entity_end'], 'entity_type_b': row_b['entity_type'], 'aspect_b': row_b['aspect'], 'sentiment_b': row_b['sentiment'], 'confidence_b': row_b['confidence']}}
                results.append((result_index, result_row))

    # Create DataFrame from results
    result_df = pd.DataFrame([x[1] for x in results], index=pd.MultiIndex.from_tuples([x[0] for x in results], names=['quote_id', 'aspect_number', 'entity_start_min', 'entity_end_max']))
    quote_to_paragraph = df_a.reset_index()[["quote_id", "paragraph_text"]].set_index("quote_id").drop_duplicates()
    result_df = result_df.merge(quote_to_paragraph, left_index=True, right_index=True, how="left")
    result_df["entity"] = result_df.apply(get_entity_from_span, axis=1)
    
    return result_df



def encode_columns_as_categoricals(df):

    for col in df.columns:
        if col not in ["entity", "entity_start_a", "entity_start_b", "entity_end_a", "entity_end_b"]:
            df[col] = pd.Categorical(df[col])
            df[col] = df[col].cat.codes.astype(int)
            # df[col] = df[col].replace(-1, pd.NA)
    return df


def assign_full_value(row, annotator):

    final_value = ''
    for column in ["entity_type", "aspect", "sentiment"]:
        if row[f"{column}_{annotator}"] == -1:
            return -1
        else:
            final_value += str(int(row[f"{column}_{annotator}"]))
    return int(final_value)

def prepare_comparison_table_for_f1_scores(comparison_table):
    encoded_table = encode_columns_as_categoricals(comparison_table)
    encoded_table["full_a"] = encoded_table.apply(assign_full_value, args=("a"), axis=1)
    encoded_table["full_b"] = encoded_table.apply(assign_full_value, args=("b"), axis=1)
    return encoded_table


def calculate_f1_score(table, column):
    table = table.reset_index()
    # Initialize TP, FP, and FN counts
    tp_gold = tp_prediction = fp_gold = fp_prediction = fn_gold = fn_prediction = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    # Compare annotations for each interval
    for (interval, gold_label, prediction_label) in zip(table['quote_id'], table[f"{column}_a"], table[f"{column}_b"]):
        if gold_label == prediction_label:
            tp_gold += 1
            tp_prediction += 1
            tp += 1
        elif gold_label != -1 and prediction_label == -1:
            fp_gold += 1
            fn_prediction += 1
            fn += 1
        elif gold_label == -1 and prediction_label != -1:
            fn_gold += 1
            fp_prediction += 1
            fp += 1
        elif gold_label != prediction_label:
            fn_gold += 1
            fp_gold += 1
            fn_prediction += 1
            fp_prediction += 1
            
    # Calculate precision, recall, and F1 score for gold
    precision_gold = tp_gold / (tp_gold + fp_gold) if (tp_gold + fp_gold) > 0 else 0
    recall_gold = tp_gold / (tp_gold + fn_gold) if (tp_gold + fn_gold) > 0 else 0
    f1_gold = 2 * (precision_gold * recall_gold) / (precision_gold + recall_gold) if (precision_gold + recall_gold) > 0 else 0
    # Calculate precision, recall, and F1 score for prediction
    precision_prediction = tp_prediction / (tp_prediction + fp_prediction) if (tp_prediction + fp_prediction) > 0 else 0
    recall_prediction = tp_prediction / (tp_prediction + fn_prediction) if (tp_prediction + fn_prediction) > 0 else 0
    f1_prediction = 2 * (precision_prediction * recall_prediction) / (precision_prediction + recall_prediction) if (precision_prediction + recall_prediction) > 0 else 0
    # Calculate overall F1 score (avergolde of gold's and prediction's F1 scores)
    overall_f1 = (f1_gold + f1_prediction) / 2
    return overall_f1




def weighted_f1_score(y_true, y_pred):
    # Calculate precision, recall, and F1 score for each class
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    # Calculate macro F1 scor
    
    return weighted_f1



def print_all_metrics(comparison_table):
    comparison_table = prepare_comparison_table_for_f1_scores(comparison_table)
    for category in ["entity_type", "aspect", "sentiment", "full"]:
        print(f"{category} F1 score: {calculate_f1_score(comparison_table, category)}")
        weighted_category_f1 = weighted_f1_score(comparison_table[f"{category}_a"], comparison_table[f"{category}_b"])
        macro_category_f1 = f1_score(comparison_table[f"{category}_a"], comparison_table[f"{category}_b"], average="macro")
        micro_category_f1 = f1_score(comparison_table[f"{category}_a"], comparison_table[f"{category}_b"], average="micro")
        weighted_category_precision = precision_score(comparison_table[f"{category}_a"], comparison_table[f"{category}_b"], average="weighted")
        weighted_category_recall = recall_score(comparison_table[f"{category}_a"], comparison_table[f"{category}_b"], average="weighted")
        print(f"{category} Weighted F1 score: {weighted_category_f1}")
        print(f"{category} Micro F1 score: {micro_category_f1}")
        print(f"{category} Macro F1 score: {macro_category_f1}")
        print(f"{category} Recall: {weighted_category_recall}")
        print(f"{category} Precision: {weighted_category_precision}")

def open_and_format_annotation_results(path) -> pd.DataFrame:
    with open(path, "r") as f:
        results_dict = json.load(f)

    results_df = create_results_df(results_dict)
    formatted_df = format_dfs(results_df)
    return formatted_df



annotation_directory = here / "cikm"


batch_results = {}
# batch: [(annotator, file), ...]
for result_file in annotation_directory.iterdir():
    annotator = result_file.stem.split("-")[-1]
    batch = "-".join(result_file.stem.split("-")[:-1])
    if batch in batch_results:
        batch_results[batch].append((annotator, result_file))
    else:
        batch_results[batch] = [(annotator, result_file)]

# ananlyze for each batch
if __name__ == "__main__":
    for batch in batch_results:
        print(f"\n\nBatch: {batch}")
        
        batch_versions_list = batch_results[batch]
        number_of_versions = len(batch_versions_list)

        for i in range(number_of_versions):
            for j in range(i+1, number_of_versions):
                annotator_i, result_file_i = batch_versions_list[i]
                annotator_j, result_file_j = batch_versions_list[j]
                result_df_i = open_and_format_annotation_results(result_file_i)
                result_df_j = open_and_format_annotation_results(result_file_j)
                comparison_table = create_comparison_table(result_df_i, result_df_j)
                print(f"\nComparing {annotator_i} ({len(result_df_i)} annotations) and {annotator_j} ({len(result_df_j)} annotations)")
                print_all_metrics(comparison_table)

