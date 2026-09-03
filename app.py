# simply run the following command to launch / open streamlit app on browser window 
# python -m streamlit run app.py

# ============================================================
# PA1 riddle-answer compatibility classifier
# streamlit inference-only demonstration
# ============================================================

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from scipy.sparse import (
    hstack,
    csr_matrix
)


# configuration
APP_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

ARTIFACT_DIR = (
    APP_DIR
    / "artifacts"
)

MODEL_PATH = (
    ARTIFACT_DIR
    / "pa1_frozen_model.keras"
)

PREPROCESSING_BUNDLE_PATH = (
    ARTIFACT_DIR
    / "pa1_preprocessing_bundle.joblib"
)

THRESHOLD_PATH = (
    ARTIFACT_DIR
    / "decision_threshold.json"
)

METADATA_PATH = (
    ARTIFACT_DIR
    / "model_metadata.json"
)


# basic text helpers
def normalise_text(text):

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def construct_input_text(
    riddle,
    candidate_answer
):

    return (
        f"{normalise_text(riddle)} "
        f"[SEP] "
        f"{normalise_text(candidate_answer)}"
    )


def tokenise_words(text):

    return re.findall(
        r"[a-z0-9]+",
        str(text).lower()
    )


def character_ngrams(
    text,
    n=3
):

    text = re.sub(
        r"\s+",
        " ",
        str(text).lower()
    ).strip()

    if len(text) < n:
        return set()

    return {
        text[
            i:i + n
        ]
        for i in range(
            len(text) - n + 1
        )
    }


def safe_divide(
    numerator,
    denominator
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


# pair-aware engineered features
# must reproduce Section 8 exactly

def compute_overlap_features(
    riddles,
    candidate_answers
):

    rows = []

    for riddle, answer in zip(
        riddles,
        candidate_answers
    ):

        riddle = str(
            riddle
        )

        answer = str(
            answer
        )

        riddle_tokens = set(
            tokenise_words(
                riddle
            )
        )

        answer_tokens = set(
            tokenise_words(
                answer
            )
        )

        word_intersection = (
            riddle_tokens
            .intersection(
                answer_tokens
            )
        )

        word_union = (
            riddle_tokens
            .union(
                answer_tokens
            )
        )

        riddle_char_ngrams = (
            character_ngrams(
                riddle,
                n=3
            )
        )

        answer_char_ngrams = (
            character_ngrams(
                answer,
                n=3
            )
        )

        char_intersection = (
            riddle_char_ngrams
            .intersection(
                answer_char_ngrams
            )
        )

        char_union = (
            riddle_char_ngrams
            .union(
                answer_char_ngrams
            )
        )

        riddle_word_count = len(
            tokenise_words(
                riddle
            )
        )

        answer_word_count = len(
            tokenise_words(
                answer
            )
        )

        riddle_char_count = len(
            riddle
        )

        answer_char_count = len(
            answer
        )

        rows.append({
            "word_jaccard":
                safe_divide(
                    len(
                        word_intersection
                    ),
                    len(
                        word_union
                    )
                ),

            "answer_word_coverage":
                safe_divide(
                    len(
                        word_intersection
                    ),
                    len(
                        answer_tokens
                    )
                ),

            "riddle_word_coverage":
                safe_divide(
                    len(
                        word_intersection
                    ),
                    len(
                        riddle_tokens
                    )
                ),

            "char_trigram_jaccard":
                safe_divide(
                    len(
                        char_intersection
                    ),
                    len(
                        char_union
                    )
                ),

            "riddle_word_count":
                riddle_word_count,

            "answer_word_count":
                answer_word_count,

            "riddle_char_count":
                riddle_char_count,

            "answer_char_count":
                answer_char_count,

            "answer_to_riddle_word_ratio":
                safe_divide(
                    answer_word_count,
                    riddle_word_count
                ),

            "answer_to_riddle_char_ratio":
                safe_divide(
                    answer_char_count,
                    riddle_char_count
                ),

            "riddle_has_question_mark":
                int(
                    "?" in riddle
                ),

            "answer_has_question_mark":
                int(
                    "?" in answer
                ),

            "answer_is_single_word":
                int(
                    answer_word_count
                    == 1
                )
        })

    return pd.DataFrame(
        rows
    )


# artifact loading
def check_required_files():

    required_files = {
        "Keras model":
            MODEL_PATH,

        "Preprocessing bundle":
            PREPROCESSING_BUNDLE_PATH,

        "Decision threshold":
            THRESHOLD_PATH,

        "Model metadata":
            METADATA_PATH
    }

    missing_files = [
        f"{name}: {path}"
        for name, path
        in required_files.items()
        if not path.exists()
    ]

    if missing_files:

        raise FileNotFoundError(
            "Missing Streamlit deployment files:\n"
            +
            "\n".join(
                missing_files
            )
        )


@st.cache_resource
def load_artifacts():

    check_required_files()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    preprocessing_bundle = (
        joblib.load(
            PREPROCESSING_BUNDLE_PATH
        )
    )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        threshold_data = json.load(
            file
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        metadata = json.load(
            file
        )

    threshold = float(
        threshold_data[
            "threshold"
        ]
    )

    return (
        model,
        preprocessing_bundle,
        threshold,
        metadata
    )


# exact pair-aware inference transformation
def build_pair_aware_features(
    preprocessing_bundle,
    riddles,
    candidate_answers
):

    combined_vectorizer = (
        preprocessing_bundle[
            "combined_vectorizer"
        ]
    )

    riddle_vectorizer = (
        preprocessing_bundle[
            "riddle_vectorizer"
        ]
    )

    answer_vectorizer = (
        preprocessing_bundle[
            "answer_vectorizer"
        ]
    )

    similarity_vectorizer = (
        preprocessing_bundle[
            "similarity_vectorizer"
        ]
    )

    numeric_scaler = (
        preprocessing_bundle[
            "numeric_scaler"
        ]
    )

    numeric_feature_names = (
        preprocessing_bundle[
            "numeric_feature_names"
        ]
    )


    riddles = [
        normalise_text(
            value
        )
        for value
        in riddles
    ]

    candidate_answers = [
        normalise_text(
            value
        )
        for value
        in candidate_answers
    ]


    input_texts = [
        construct_input_text(
            riddle,
            answer
        )
        for riddle, answer
        in zip(
            riddles,
            candidate_answers
        )
    ]


    # original combined TF-IDF representation
    X_combined = (
        combined_vectorizer
        .transform(
            input_texts
        )
    )


    # separate riddle and answer TF-IDF
    X_riddle = (
        riddle_vectorizer
        .transform(
            riddles
        )
    )

    X_answer = (
        answer_vectorizer
        .transform(
            candidate_answers
        )
    )


    # engineered overlap/structural features
    overlap_features = (
        compute_overlap_features(
            riddles,
            candidate_answers
        )
    )


    # TF-IDF cosine similarity
    similarity_riddle_matrix = (
        similarity_vectorizer
        .transform(
            riddles
        )
    )

    similarity_answer_matrix = (
        similarity_vectorizer
        .transform(
            candidate_answers
        )
    )

    cosine_values = np.asarray(
        similarity_riddle_matrix
        .multiply(
            similarity_answer_matrix
        )
        .sum(
            axis=1
        )
    ).ravel()


    overlap_features[
        "tfidf_cosine_similarity"
    ] = cosine_values


    # enforce exact training feature order
    overlap_features = (
        overlap_features[
            numeric_feature_names
        ]
    )


    numeric_scaled = (
        numeric_scaler
        .transform(
            overlap_features
            .to_numpy()
        )
    )


    # final pair-aware matrix
    X_pair_aware_sparse = hstack([
        csr_matrix(
            X_combined
        ),
        X_riddle,
        X_answer,
        csr_matrix(
            numeric_scaled
        )
    ])


    X_pair_aware = (
        X_pair_aware_sparse
        .toarray()
        .astype(
            "float32"
        )
    )


    return (
        input_texts,
        X_pair_aware,
        overlap_features
    )


# deployment integrity
def validate_deployment(
    model,
    preprocessing_bundle,
    threshold,
    metadata
):

    (
        _,
        sample_X,
        _
    ) = build_pair_aware_features(
        preprocessing_bundle,
        riddles=[
            "What has keys but cannot open locks?"
        ],
        candidate_answers=[
            "A piano"
        ]
    )


    feature_dim = int(
        sample_X.shape[1]
    )

    model_dim = int(
        model.input_shape[-1]
    )

    metadata_dim = int(
        metadata[
            "input_dimension"
        ]
    )


    if not (
        feature_dim
        == model_dim
        == metadata_dim
    ):

        raise ValueError(
            "Deployment feature-dimension mismatch: "
            f"constructed={feature_dim}, "
            f"model={model_dim}, "
            f"metadata={metadata_dim}"
        )


    if not np.isclose(
        threshold,
        float(
            metadata[
                "decision_threshold"
            ]
        )
    ):

        raise ValueError(
            "Deployment threshold metadata mismatch."
        )


# prediction
def predict_batch(
    model,
    preprocessing_bundle,
    riddles,
    candidate_answers,
    threshold
):

    (
        input_texts,
        X,
        engineered_features
    ) = build_pair_aware_features(
        preprocessing_bundle,
        riddles,
        candidate_answers
    )


    probabilities = (
        model.predict(
            X,
            verbose=0
        )
        .ravel()
    )


    predicted_labels = (
        probabilities
        >= threshold
    ).astype(int)


    predicted_classes = np.where(
        predicted_labels == 1,
        "correct",
        "incorrect"
    )


    return (
        input_texts,
        probabilities,
        predicted_labels,
        predicted_classes,
        engineered_features
    )


# streamlit web interface
st.set_page_config(
    page_title=(
        "Riddle–Answer "
        "Compatibility Classifier"
    ),
    page_icon="🧠",
    layout="wide"
)


st.title(
    "Riddle–Answer Pair Compatibility Classifier"
)


st.caption(
    "CM3015 Machine Learning and Neural Networks — "
    "PA1 inference-only demonstration"
)


try:

    (
        model,
        preprocessing_bundle,
        threshold,
        metadata
    ) = load_artifacts()

    validate_deployment(
        model,
        preprocessing_bundle,
        threshold,
        metadata
    )

except Exception as error:

    st.error(
        "The frozen PA1 deployment artifacts "
        "could not be loaded."
    )

    st.exception(
        error
    )

    st.stop()


# sidebar
st.sidebar.header(
    "Frozen Model"
)

st.sidebar.write(
    f"Experiment: "
    f"{metadata.get('selected_experiment', 'PA1')}"
)

st.sidebar.write(
    "Architecture: 128 → 64 Dense"
)

st.sidebar.write(
    "Dropout: 0.50"
)

st.sidebar.write(
    "Learning rate: 5e-4"
)

st.sidebar.write(
    "Input dimension: "
    f"{metadata.get('input_dimension', 'unknown')}"
)

st.sidebar.write(
    f"Decision threshold: {threshold:.2f}"
)

st.sidebar.info(
    "The model and threshold are frozen. "
    "This application performs inference only."
)


# single prediction
st.header(
    "Single Prediction"
)

riddle_input = st.text_area(
    "Riddle",
    placeholder=(
        "Enter a riddle..."
    )
)

answer_input = st.text_input(
    "Candidate answer",
    placeholder=(
        "Enter a candidate answer..."
    )
)


if st.button(
    "Classify pair"
):

    if not riddle_input.strip():

        st.warning(
            "Please enter a riddle."
        )

    elif not answer_input.strip():

        st.warning(
            "Please enter a candidate answer."
        )

    else:

        (
            input_texts,
            probabilities,
            predicted_labels,
            predicted_classes,
            engineered_features
        ) = predict_batch(
            model=model,
            preprocessing_bundle=
                preprocessing_bundle,
            riddles=[
                riddle_input
            ],
            candidate_answers=[
                answer_input
            ],
            threshold=threshold
        )


        probability = float(
            probabilities[0]
        )

        predicted_class = (
            predicted_classes[0]
        )


        st.metric(
            "Probability of correct match",
            f"{probability:.4f}"
        )


        st.metric(
            "Frozen decision threshold",
            f"{threshold:.2f}"
        )


        if predicted_class == "correct":

            st.success(
                "Prediction: CORRECT riddle–answer pair"
            )

        else:

            st.error(
                "Prediction: INCORRECT riddle–answer pair"
            )


        with st.expander(
            "Engineered compatibility features"
        ):

            st.dataframe(
                engineered_features
            )


# batch CSV prediction
st.header(
    "Batch CSV Prediction"
)

st.write(
    "Upload a CSV containing the columns "
    "`riddle` and `candidate_answer`."
)


uploaded_file = st.file_uploader(
    "Choose CSV file",
    type=[
        "csv"
    ]
)


if uploaded_file is not None:

    try:

        batch_df = pd.read_csv(
            uploaded_file
        )

    except Exception as error:

        st.error(
            "The CSV file could not be read."
        )

        st.exception(
            error
        )

        st.stop()


    required_columns = {
        "riddle",
        "candidate_answer"
    }

    missing_columns = (
        required_columns
        - set(
            batch_df.columns
        )
    )


    if missing_columns:

        st.error(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    elif batch_df.empty:

        st.warning(
            "The uploaded CSV file is empty."
        )

    else:

        batch_df = (
            batch_df.copy()
        )

        batch_df[
            "riddle"
        ] = (
            batch_df[
                "riddle"
            ]
            .fillna("")
            .astype(str)
        )

        batch_df[
            "candidate_answer"
        ] = (
            batch_df[
                "candidate_answer"
            ]
            .fillna("")
            .astype(str)
        )


        valid_rows = (
            batch_df[
                "riddle"
            ]
            .str.strip()
            .ne("")
            &
            batch_df[
                "candidate_answer"
            ]
            .str.strip()
            .ne("")
        )


        if (
            valid_rows.sum()
            == 0
        ):

            st.warning(
                "No valid rows contain both "
                "a riddle and candidate answer."
            )

        else:

            valid_batch_df = (
                batch_df[
                    valid_rows
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )


            (
                input_texts,
                probabilities,
                predicted_labels,
                predicted_classes,
                engineered_features
            ) = predict_batch(
                model=model,
                preprocessing_bundle=
                    preprocessing_bundle,
                riddles=
                    valid_batch_df[
                        "riddle"
                    ],
                candidate_answers=
                    valid_batch_df[
                        "candidate_answer"
                    ],
                threshold=threshold
            )


            result_df = (
                valid_batch_df
                .copy()
            )

            result_df[
                "probability_correct"
            ] = probabilities

            result_df[
                "predicted_label"
            ] = predicted_labels

            result_df[
                "predicted_class"
            ] = predicted_classes


            result_with_features_df = (
                pd.concat(
                    [
                        result_df,
                        engineered_features
                        .reset_index(
                            drop=True
                        )
                    ],
                    axis=1
                )
            )


            st.subheader(
                "Batch Prediction Results"
            )

            st.dataframe(
                result_df,
                use_container_width=True
            )


            with st.expander(
                "Results with engineered features"
            ):

                st.dataframe(
                    result_with_features_df,
                    use_container_width=True
                )


            csv_output = (
                result_with_features_df
                .to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )


            st.download_button(
                label=(
                    "Download predictions as CSV"
                ),
                data=csv_output,
                file_name=(
                    "pa1_riddle_answer_predictions.csv"
                ),
                mime="text/csv"
            )


# final model information
with st.expander(
    "Frozen model metadata"
):

    st.json(
        metadata
    )
