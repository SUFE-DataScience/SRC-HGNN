import pandas as pd
import re
import string
from transformers import pipeline
import nltk
from nltk.corpus import stopwords
import json

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

file_path = r"D:\PyCharm\data\data.xlsx"

df = pd.read_excel(file_path)

sentiment_pipe = pipeline("text-classification", model="VictorSanh/roberta-base-finetuned-yelp-polarity")
tokenizer = sentiment_pipe.tokenizer

label_mapping = {
    "LABEL_0": "NEGATIVE",
    "LABEL_1": "POSITIVE"
}

def preprocess(text):
    """
    Text preprocessing:
      - Lowercase
      - Remove punctuation
      - Remove stopwords
      - Normalize whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def sliding_window_sentiment(text, pipe_obj, tokenizer_obj, max_len=512, stride=256):
    """
    For long texts (token count > model limit), use a sliding window and combine
    segment predictions via majority voting.

    To avoid exceeding the max length after adding special tokens internally,
    compute an effective limit:
      effective_max_len = max_len - special_tokens_count
    where special_tokens_count is the number of special tokens the model adds
    for a single sequence.

    Args:
      text: input text to classify
      pipe_obj: transformers pipeline for text classification
      tokenizer_obj: corresponding tokenizer for token counting/decoding
      max_len: maximum token length supported by the model (default 512)
      stride: window stride (default 256) to ensure overlap
    """
    special_tokens_count = len(tokenizer_obj.encode("", add_special_tokens=True))
    effective_max_len = max_len - special_tokens_count

    tokens = tokenizer_obj.encode(text, add_special_tokens=False)

    if len(tokens) <= effective_max_len:
        result = pipe_obj(text)[0]
        result["label"] = label_mapping.get(result["label"], result["label"])
        return result
    else:
        results = []
        for i in range(0, len(tokens), stride):
            chunk_tokens = tokens[i:i + effective_max_len]
            if not chunk_tokens:
                break

            chunk_text = tokenizer_obj.decode(
                chunk_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            if not chunk_text.strip():
                continue

            result = pipe_obj(chunk_text)[0]
            result["label"] = label_mapping.get(result["label"], result["label"])
            results.append(result)

            if i + effective_max_len >= len(tokens):
                break

        if not results:
            return {"label": "UNKNOWN", "score": 0.0, "details": []}

        votes = {"POSITIVE": 0, "NEGATIVE": 0}
        score_sum = {"POSITIVE": 0.0, "NEGATIVE": 0.0}
        for res in results:
            label = res["label"]
            score = res["score"]
            votes[label] += 1
            score_sum[label] += score

        if votes["POSITIVE"] > votes["NEGATIVE"]:
            overall_label = "POSITIVE"
            overall_score = score_sum["POSITIVE"] / votes["POSITIVE"]
        elif votes["NEGATIVE"] > votes["POSITIVE"]:
            overall_label = "NEGATIVE"
            overall_score = score_sum["NEGATIVE"] / votes["NEGATIVE"]
        else:
            if score_sum["POSITIVE"] >= score_sum["NEGATIVE"]:
                overall_label = "POSITIVE"
                overall_score = score_sum["POSITIVE"] / votes["POSITIVE"] if votes["POSITIVE"] > 0 else 0.0
            else:
                overall_label = "NEGATIVE"
                overall_score = score_sum["NEGATIVE"] / votes["NEGATIVE"] if votes["NEGATIVE"] > 0 else 0.0

        return {"label": overall_label, "score": overall_score, "details": results}

if __name__ == "__main__":
    output_data = []
    for idx, row in df.iterrows():
        review_text = row.get("Review_Text")
        if not isinstance(review_text, str) or not review_text.strip():
            print(f"Review {idx}: no valid text")
            continue

        preprocessed_text = preprocess(review_text)
        sentiment_result = sliding_window_sentiment(preprocessed_text, sentiment_pipe, tokenizer)
        print(f"Review {idx}: {sentiment_result}")

        details_str = json.dumps(sentiment_result.get("details", []), ensure_ascii=True)

        output_data.append({
            "Review_ID": idx,
            "Review_Text": review_text,
            "Preprocessed_Text": preprocessed_text,
            "Sentiment_Label": sentiment_result.get("label", "UNKNOWN"),
            "Sentiment_Score": sentiment_result.get("score", 0.0),
            "Details": details_str
        })

    output_df = pd.DataFrame(output_data)
    output_csv_path = r"sentiment_results.csv"
    output_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    print(f"Results written to CSV file: {output_csv_path}")
